"""
Post-processing di mappe di bordo prodotte da una CNN.

    immagine originale   -> g, theta
    probabilita' CNN     -> bordi candidati (isteresi)
    candidati + g + theta -> bordi collegati (linking + bridging)
    bordi collegati      -> contorni -> visualizzazione

API e formato di output identici alla versione precedente:
    link_cnn_edges() restituisce {"g", "theta", "gx", "gy", "linked_edges"}
    linked_edges e' uint8 con valori {0, 1}
    extract_contours() restituisce una lista di contorni OpenCV

Dipendenze fra i dati:
    g      -> usato solo da orientation_aware_linking (soglia min_gradient)
    theta  -> usato da orientation_aware_linking e da bridge_small_gaps
    gx, gy -> non usati internamente, esportati solo per ispezione
"""

import cv2
import numpy as np

# ==============================================================================
# PARAMETRI DI DEFAULT E CONFIGURAZIONE
# ==============================================================================
DEFAULT_SIGMA = 1.6

# Normalizzazione della magnitudine del gradiente.
# 100.0 = divisione per il massimo (comportamento originale).
# Valori come 99.0 rendono min_gradient insensibile a singoli outlier
# (riflessi speculari) al costo di un clipping in [0, 1].
DEFAULT_GRADIENT_PERCENTILE = 100.0

DEFAULT_LOW_PROB = 0.20
DEFAULT_HIGH_PROB = 0.50
DEFAULT_MIN_GRADIENT = 0.05
DEFAULT_MAX_ANGLE_DIFF_DEG = 35.0

DEFAULT_BRIDGE_GAPS = True

# max_gap e' una distanza EUCLIDEA fra i due estremi da collegare.
# Due componenti distinte non possono essere 8-adiacenti, quindi la
# distanza minima possibile e' 2: max_gap = N chiude buchi di N-1 pixel.
DEFAULT_MAX_GAP = 20
DEFAULT_SEARCH_ANGLE_DEG = 25.0
DEFAULT_BRIDGE_MAX_ANGLE_DIFF_DEG = 30.0
DEFAULT_ONE_LINK_PER_PAIR = True

DEFAULT_MIN_AREA = 10.0
DEFAULT_CONTOUR_THICKNESS = 2
# ==============================================================================


# ------------------------------------------------------------------------------
# Utilita' angolari
# ------------------------------------------------------------------------------
def angle_diff(a, b):
    """
    Differenza fra ORIENTAMENTI, pi-periodica: risultato in [0, pi/2].

    Da usare per confrontare tangenti o normali, il cui verso e' arbitrario.
    Accetta scalari o array.
    """
    d = np.abs(np.asarray(a, dtype=np.float64)
               - np.asarray(b, dtype=np.float64)) % np.pi
    return np.minimum(d, np.pi - d)


def dir_diff(a, b):
    """
    Differenza fra DIREZIONI con verso, 2pi-periodica: risultato in [0, pi].

    Da usare per il cono di ricerca del bridging: distingue "avanti" da
    "indietro", cosa che angle_diff non puo' fare.
    Accetta scalari o array.
    """
    d = (np.asarray(a, dtype=np.float64)
         - np.asarray(b, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi
    return np.abs(d)


class _UnionFind:
    """Union-find compatto, usato per non duplicare i ponti fra componenti."""

    __slots__ = ("parent",)

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        root = a
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[a] != root:
            self.parent[a], a = root, self.parent[a]
        return root

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[rb] = ra
        return True


# ------------------------------------------------------------------------------
# Gradienti
# ------------------------------------------------------------------------------
def gaussian_blur(gray: np.ndarray, sigma: float = DEFAULT_SIGMA) -> np.ndarray:
    k = max(3, int(6 * sigma + 1) | 1)
    return cv2.GaussianBlur(gray, (k, k), sigmaX=sigma, sigmaY=sigma)


def gradients(
    gray: np.ndarray,
    sigma: float = DEFAULT_SIGMA,
    norm_percentile: float = DEFAULT_GRADIENT_PERCENTILE,
):
    """
    Magnitudine g e orientazione theta calcolate sull'immagine ORIGINALE.

    theta = arctan2(gy, gx) e' la direzione del gradiente, cioe' la NORMALE
    al bordo. La tangente al bordo e' theta + pi/2.

    Convenzione di assi: x verso destra (colonne), y verso il basso (righe).
    La stessa usata da arctan2(dy, dx) nel bridging: sono coerenti.
    """
    if gray.ndim != 2:
        raise ValueError("gray deve essere HxW")

    img = gaussian_blur(gray, sigma).astype(np.float32) / 255.0

    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

    g = np.sqrt(gx * gx + gy * gy)

    if norm_percentile >= 100.0:
        ref = float(g.max())
        g /= (ref + 1e-8)
    else:
        ref = float(np.percentile(g, norm_percentile))
        g /= (ref + 1e-8)
        np.clip(g, 0.0, 1.0, out=g)

    theta = np.arctan2(gy, gx)

    return g, theta, gx, gy


# ------------------------------------------------------------------------------
# Stadio 1: isteresi guidata dall'orientamento
# ------------------------------------------------------------------------------
def orientation_aware_linking(
    edge_prob: np.ndarray,
    g: np.ndarray,
    theta: np.ndarray,
    low_prob: float = DEFAULT_LOW_PROB,
    high_prob: float = DEFAULT_HIGH_PROB,
    min_gradient: float = DEFAULT_MIN_GRADIENT,
    max_angle_diff_deg: float = DEFAULT_MAX_ANGLE_DIFF_DEG,
    require_source_gradient: bool = False,
) -> np.ndarray:
    """
    Combina l'uscita della CNN con l'informazione geometrica.

    Un pixel e':
      - forte se probabilita' >= high_prob
      - debole se low_prob <= probabilita' < high_prob

    Un pixel debole viene accettato solo se:
      1. e' 8-connesso a un pixel gia' accettato
      2. la sua magnitudine di gradiente e' >= min_gradient
      3. il suo theta e' compatibile con quello del vicino accettato

    require_source_gradient=True estende il controllo 2 anche al pixel
    sorgente: evita che un pixel forte con gradiente quasi nullo propaghi un
    theta che e' puro rumore. Default False per non alterare il
    comportamento preesistente.
    """
    if edge_prob.ndim != 2:
        raise ValueError("edge_prob must be HxW")
    if g.shape != edge_prob.shape or theta.shape != edge_prob.shape:
        raise ValueError("edge_prob, g and theta must have the same shape")
    if not (0.0 <= low_prob <= high_prob):
        raise ValueError("richiesto 0 <= low_prob <= high_prob")
    if edge_prob.dtype == np.uint8:
        raise ValueError(
            "edge_prob deve essere float in [0, 1], non uint8 in [0, 255]"
        )

    prob = np.asarray(edge_prob, dtype=np.float32)

    strong = prob >= high_prob
    weak = (prob >= low_prob) & ~strong

    accepted = strong.copy()
    h, w = prob.shape

    max_diff = np.deg2rad(max_angle_diff_deg)

    # Precalcolo: un pixel debole non e' mai accettabile se il suo
    # gradiente e' sotto soglia. Filtrarlo qui evita un test nel loop.
    weak &= (g >= min_gradient)

    theta_f = np.asarray(theta, dtype=np.float64)
    g_f = np.asarray(g, dtype=np.float32)

    stack = [(int(y), int(x)) for y, x in np.argwhere(strong)]

    while stack:
        y, x = stack.pop()

        if require_source_gradient and g_f[y, x] < min_gradient:
            continue

        theta_src = theta_f[y, x]

        y0 = y - 1 if y > 0 else 0
        y1 = y + 2 if y + 2 <= h else h
        x0 = x - 1 if x > 0 else 0
        x1 = x + 2 if x + 2 <= w else w

        for yy in range(y0, y1):
            for xx in range(x0, x1):

                if yy == y and xx == x:
                    continue

                if accepted[yy, xx]:
                    continue

                if not weak[yy, xx]:
                    continue

                d = abs(theta_src - theta_f[yy, xx]) % np.pi
                if min(d, np.pi - d) > max_diff:
                    continue

                accepted[yy, xx] = True
                stack.append((yy, xx))

    return accepted.astype(np.uint8)


# ------------------------------------------------------------------------------
# Stadio 2: chiusura dei piccoli gap
# ------------------------------------------------------------------------------
def bridge_small_gaps(
    edges: np.ndarray,
    theta: np.ndarray,
    max_gap: int = DEFAULT_MAX_GAP,
    search_angle_deg: float = DEFAULT_SEARCH_ANGLE_DEG,
    max_angle_diff_deg: float = DEFAULT_BRIDGE_MAX_ANGLE_DIFF_DEG,
    one_link_per_pair: bool = DEFAULT_ONE_LINK_PER_PAIR,
) -> np.ndarray:
    """
    Chiude piccoli gap cercando estremi compatibili dentro un cono angolare
    orientato lungo la tangente del bordo.

    Per ogni pixel di bordo, e per ciascuno dei due VERSI della tangente:
      1. cerca pixel di bordo entro max_gap (distanza euclidea)
      2. scarta i pixel della PROPRIA componente connessa
      3. accetta solo candidati entro +/- search_angle_deg dal verso di
         ricerca, misurato con dir_diff (2pi-periodica)
      4. verifica che la tangente del candidato sia compatibile con la
         direzione della connessione, entro max_angle_diff_deg
      5. tiene il candidato compatibile piu' vicino

    I ponti raccolti vengono poi applicati in ordine di distanza crescente.
    Con one_link_per_pair=True due componenti gia' collegate (direttamente o
    per transitivita') non vengono ricollegate: evita fasci di ponti
    ridondanti fra gli stessi due tratti.

    Differenze rispetto alla versione precedente, tutte necessarie perche' la
    funzione facesse davvero qualcosa:
      - il cono usa dir_diff invece di angle_diff, altrimenti i due versi
        sono indistinguibili e la ricerca guarda anche all'indietro
      - il vincolo di componente sostituisce il test "dist < 2.0": senza di
        esso il candidato piu' vicino e' quasi sempre il prolungamento
        stesso del bordo, e il gap non viene mai chiuso
      - il test di coerenza sulla tangente iniziale e' rimosso perche'
        implicato dal cono, che e' piu' stretto
    """
    if edges.shape != theta.shape:
        raise ValueError("edges e theta devono avere la stessa forma")
    if max_gap < 2:
        return (edges > 0).astype(np.uint8)

    src = (edges > 0).astype(np.uint8)
    out = src.copy()
    h, w = src.shape

    n_lab, lab = cv2.connectedComponents(src, connectivity=8)
    if n_lab <= 2:                      # sfondo + al massimo una componente
        return out

    search_angle = np.deg2rad(search_angle_deg)
    max_diff = np.deg2rad(max_angle_diff_deg)
    max_gap_sq = float(max_gap) ** 2

    theta_f = np.asarray(theta, dtype=np.float64)
    ys, xs = np.nonzero(src)

    proposals = []

    for y, x in zip(ys.tolist(), xs.tolist()):

        # theta = normale al bordo, tangent = direzione del bordo
        tangent_start = theta_f[y, x] + np.pi / 2.0
        label_src = int(lab[y, x])

        y0 = max(0, y - max_gap)
        y1 = min(h, y + max_gap + 1)
        x0 = max(0, x - max_gap)
        x1 = min(w, x + max_gap + 1)

        for turn in (0.0, np.pi):

            search_direction = tangent_start + turn

            best_candidate = None
            best_distance_sq = np.inf

            for yy in range(y0, y1):
                for xx in range(x0, x1):

                    if not src[yy, xx]:
                        continue

                    # Stessa componente: gia' collegati, niente da fare.
                    # Include il pixel stesso e i suoi 8 vicini.
                    if int(lab[yy, xx]) == label_src:
                        continue

                    dy = yy - y
                    dx = xx - x
                    dist_sq = float(dx * dx + dy * dy)

                    if dist_sq > max_gap_sq:
                        continue

                    if dist_sq >= best_distance_sq:
                        continue

                    connection_angle = np.arctan2(dy, dx)

                    # 1. il candidato deve stare nel cono, con il verso giusto
                    d = (search_direction - connection_angle + np.pi) \
                        % (2.0 * np.pi) - np.pi
                    if abs(d) > search_angle:
                        continue

                    # 2. coerenza della tangente del candidato con il ponte
                    tangent_end = theta_f[yy, xx] + np.pi / 2.0
                    d = abs(tangent_end - connection_angle) % np.pi
                    if min(d, np.pi - d) > max_diff:
                        continue

                    best_distance_sq = dist_sq
                    best_candidate = (yy, xx)

            if best_candidate is not None:
                yy, xx = best_candidate
                proposals.append(
                    (best_distance_sq, y, x, yy, xx)
                )

    # Applicazione: ponti piu' corti per primi, ordine deterministico.
    proposals.sort()

    uf = _UnionFind(n_lab)

    for _, y, x, yy, xx in proposals:
        if one_link_per_pair:
            if not uf.union(int(lab[y, x]), int(lab[yy, xx])):
                continue

        cv2.line(out, (int(x), int(y)), (int(xx), int(yy)), 1, 1)

    return out


# ------------------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------------------
def link_cnn_edges(
    image_bgr: np.ndarray,
    edge_prob: np.ndarray,
    sigma: float = DEFAULT_SIGMA,
    low_prob: float = DEFAULT_LOW_PROB,
    high_prob: float = DEFAULT_HIGH_PROB,
    min_gradient: float = DEFAULT_MIN_GRADIENT,
    max_angle_diff_deg: float = DEFAULT_MAX_ANGLE_DIFF_DEG,
    bridge_gaps: bool = DEFAULT_BRIDGE_GAPS,
    max_gap: int = DEFAULT_MAX_GAP,
    bridge_search_angle_deg: float = DEFAULT_SEARCH_ANGLE_DEG,
    bridge_max_angle_diff_deg: float = DEFAULT_BRIDGE_MAX_ANGLE_DIFF_DEG,
    gradient_percentile: float = DEFAULT_GRADIENT_PERCENTILE,
    require_source_gradient: bool = False,
):
    """
    Pipeline completa di post-processing.

    I due stadi hanno ora soglie angolari separate: max_angle_diff_deg per il
    linking, bridge_max_angle_diff_deg per il bridging. Prima la seconda
    costante esisteva ma veniva sempre sovrascritta dalla prima.

    Restituisce gli stessi campi di prima.
    """
    if image_bgr.ndim == 3:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    elif image_bgr.ndim == 2:
        gray = image_bgr
    else:
        raise ValueError("image_bgr deve essere HxWx3 o HxW")

    if gray.shape != edge_prob.shape:
        raise ValueError("immagine ed edge_prob devono avere la stessa forma")

    g, theta, gx, gy = gradients(
        gray, sigma=sigma, norm_percentile=gradient_percentile
    )

    linked = orientation_aware_linking(
        edge_prob=edge_prob,
        g=g,
        theta=theta,
        low_prob=low_prob,
        high_prob=high_prob,
        min_gradient=min_gradient,
        max_angle_diff_deg=max_angle_diff_deg,
        require_source_gradient=require_source_gradient,
    )

    if bridge_gaps:
        linked = bridge_small_gaps(
            linked,
            theta,
            max_gap=max_gap,
            search_angle_deg=bridge_search_angle_deg,
            max_angle_diff_deg=bridge_max_angle_diff_deg,
        )

    return {
        "g": g,
        "theta": theta,
        "gx": gx,
        "gy": gy,
        "linked_edges": linked,
    }


# ------------------------------------------------------------------------------
# Contorni
# ------------------------------------------------------------------------------
def extract_contours(
    linked_edges: np.ndarray,
    min_area: float = DEFAULT_MIN_AREA,
    min_length: float = 0.0,
    # retrieval_mode: int = cv2.RETR_EXTERNAL, RETR_LIST
    retrieval_mode: int = cv2.RETR_LIST
    
):
    """
    Estrae i contorni dalla mappa finale di bordi.

    ATTENZIONE, il comportamento di default e' quello originale e ha due
    limiti intrinseci:

      - findContours traccia il PERIMETRO dei tratti di bordo, non le regioni
        che i bordi racchiudono. Un tratto aperto, per quanto lungo, ha area
        praticamente nulla e viene eliminato da min_area.
      - RETR_EXTERNAL scarta i contorni annidati (oggetti interni, buchi).

    Per conservare i tratti aperti: min_area=0.0 e min_length > 0, che filtra
    su cv2.arcLength invece che sull'area.
    Per le regioni annidate: retrieval_mode=cv2.RETR_CCOMP o RETR_TREE.
    Per segmentare le REGIONI delimitate dai bordi serve invece
    cv2.connectedComponents sul complemento della mappa, non questa funzione.
    """
    binary = (linked_edges > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        binary,
        retrieval_mode,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if min_area > 0.0:
        contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    if min_length > 0.0:
        contours = [
            c for c in contours
            if cv2.arcLength(c, False) >= min_length
        ]

    return contours


def color_contours(
    image_bgr: np.ndarray,
    contours,
    thickness: int = DEFAULT_CONTOUR_THICKNESS,
):
    """
    Disegna ogni contorno con uno pseudo-colore deterministico.
    """
    out = image_bgr.copy()

    for i, contour in enumerate(contours):
        color = (
            int((37 * i + 50) % 255),
            int((91 * i + 120) % 255),
            int((157 * i + 200) % 255),
        )
        cv2.drawContours(out, [contour], -1, color, thickness)

    return out

