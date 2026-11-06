"""조건 간 CAP 변화 (Advanced Science 논문의 PCA shift + t-SNE 분석 일반화).

전체 CAP 으로 PCA 를 fit -> 조건 A, B 의 CAP 을 같은 축으로 transform
-> 유전자마다 A -> B 이동 벡터 delta 로 아래 값 계산
    norm     : |delta| (얼마나 움직였나)
    centroid : delta 를 전체 이동 방향(centroid 차이)에 투영한 값
    cent_sim : delta 와 전체 이동 방향의 cosine
    PCdot    : A 점수 · B 점수
    gene_sim : A 점수와 B 점수의 cosine (자기 위치 유지 정도)
"""
import colorsys

import numpy as np
import pandas as pd
from matplotlib.colors import to_hex
from sklearn.decomposition import PCA


def cap_pca_spectrum(cap, n_show=10):
    """전체 CAP PCA 의 eigenvalue / 설명분산 (몇 개 PC 를 쓸지 판단용)"""
    p = PCA().fit(cap.values)
    k = min(n_show, len(p.explained_variance_))
    return pd.DataFrame(dict(PC=np.arange(1, k + 1),
                             eigenvalue=p.explained_variance_[:k],
                             var_ratio=p.explained_variance_ratio_[:k],
                             cum_ratio=np.cumsum(p.explained_variance_ratio_)[:k]))

def cap_shift(caps, base, target, pca_on='all', n_components=3):
    """caps   : run_cscore 결과의 caps dict
    base   : 기준 조건 라벨 (예: 'yoked' 또는 0)
    target : 비교할 조건 라벨 (예: 'trained' 또는 1). delta = target - base
    pca_on : PCA 를 fit 할 CAP (기본 전체 'all')"""
    labels = [k for k in caps if k != pca_on]
    for x in (base, target):
        if x not in caps:
            raise ValueError(f"'{x}' 라벨의 CAP 이 없어요. 가능한 라벨: {labels}")
    if base == target:
        raise ValueError("base 와 target 이 같아요")
    print(f"[shift] {target} - {base} (PCA fit: {pca_on})")
    genes = list(caps[pca_on].index)
    pca = PCA(n_components=n_components).fit(caps[pca_on].loc[genes, genes].values)
    cols = [f'PC{i + 1}' for i in range(n_components)]
    S = {k: pd.DataFrame(pca.transform(caps[k].loc[genes, genes].values), index=genes, columns=cols)
         for k in (pca_on, base, target)}
    S_base, S_target = S[base].values, S[target].values
    delta = S_target - S_base
    norm = np.linalg.norm(delta, axis=1)
    u = S_target.mean(0) - S_base.mean(0)
    u = u / np.linalg.norm(u)
    contrib = delta @ u
    with np.errstate(invalid='ignore', divide='ignore'):
        cent_sim = contrib / norm
        dot = (S_base * S_target).sum(1)
        gene_sim = dot / (np.linalg.norm(S_base, axis=1) * np.linalg.norm(S_target, axis=1))
    shift = pd.DataFrame(dict(norm=norm, centroid=contrib, PCdot=dot,
                              cent_sim=cent_sim, gene_sim=gene_sim), index=genes)
    return dict(pca=pca, scores=S, shift=shift, var_ratio=pca.explained_variance_ratio_,
                base=base, target=target, pca_on=pca_on)

def _brighten(rgb, lightness=0.4):
    h, l, s = colorsys.rgb_to_hls(*rgb)
    return colorsys.hls_to_rgb(h, max(l, lightness), s)


def rgb_colors(scores, min_brightness=0.3, brighten=True):
    """PC1-3 점수를 RGB 색으로 (논문 그림 방식). 반환: gene -> hex Series"""
    X = scores.iloc[:, :3]
    rng = (X.max() - X.min()).replace(0, 1)
    rgb = (X - X.min()) / rng
    rgb = rgb * (1 - min_brightness) + min_brightness
    f = _brighten if brighten else (lambda v: v)
    return pd.Series([to_hex(f(tuple(r))) for r in rgb.values], index=X.index)


def shift_tsne(shift, cols=('norm', 'cent_sim', 'gene_sim'), perplexity=30, random_state=5):
    """shift 지표로 t-SNE 2D. 반환: comp1, comp2 + 사용한 지표 컬럼"""
    from sklearn.manifold import TSNE
    X = shift[list(cols)].fillna(0.0)
    perplexity = min(perplexity, len(X) - 1)
    Z = TSNE(n_components=2, perplexity=perplexity, random_state=random_state).fit_transform(X.values)
    out = pd.DataFrame(Z, index=X.index, columns=['comp1', 'comp2'])
    return out.join(shift[list(cols)])



def rgb_colors(scores, min_brightness=0.3, brighten=True, invert=None):
    """PC1-3 점수를 RGB 색으로 (논문 그림 방식). 반환: gene -> hex Series
    invert: 뒤집을 채널. 예) invert='G' 또는 ['R', 'B'] (논문은 G 를 뒤집음)"""
    X = scores.iloc[:, :3]
    rng = (X.max() - X.min()).replace(0, 1)
    rgb = (X - X.min()) / rng
    if invert:
        for ch in ([invert] if isinstance(invert, str) else invert):
            i = 'RGB'.index(ch.upper())
            rgb.iloc[:, i] = 1 - rgb.iloc[:, i]
    rgb = rgb * (1 - min_brightness) + min_brightness
    f = _brighten if brighten else (lambda v: v)
    return pd.Series([to_hex(f(tuple(r))) for r in rgb.values], index=X.index)