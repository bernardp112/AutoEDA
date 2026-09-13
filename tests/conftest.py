"""
Fixtures compartilhadas entre os testes do AutoEDA.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autoeda.config import AutoEDAConfig


@pytest.fixture
def config() -> AutoEDAConfig:
    """Configuração padrão (pt-br), usada pela maioria dos testes."""
    return AutoEDAConfig()


@pytest.fixture
def config_en() -> AutoEDAConfig:
    """Configuração em inglês, para os testes de i18n."""
    return AutoEDAConfig(language="en-us")


@pytest.fixture
def simple_binary_df() -> pd.DataFrame:
    """Dataset pequeno e determinístico: target binário + 1 preditor
    numérico fortemente associado + 1 categórico fraco. Sem valores
    ausentes, sem outliers — usado como caso "feliz" de referência.
    """
    rng = np.random.default_rng(42)
    n = 300
    target = rng.choice(["sim", "nao"], size=n, p=[0.5, 0.5])
    target_bin = (target == "sim").astype(int)

    preditor_forte = np.where(
        target_bin == 1,
        rng.normal(80, 5, n),
        rng.normal(50, 5, n),
    )
    categoria_fraca = rng.choice(["X", "Y", "Z"], size=n)

    return pd.DataFrame(
        {
            "target": target,
            "preditor_forte": preditor_forte,
            "categoria_fraca": categoria_fraca,
        }
    )


@pytest.fixture
def messy_df() -> pd.DataFrame:
    """Dataset com vários problemas propositais, para exercitar cada
    regra de recomendação de uma vez: id, constante, quase-constante,
    assimetria, ausência (inclusive ligada ao target), outliers,
    correlação alta e desbalanceamento de classes.
    """
    rng = np.random.default_rng(7)
    n = 400

    target = rng.choice(["inadimplente", "adimplente"], size=n, p=[0.15, 0.85])
    x = rng.normal(0, 1, n)

    df = pd.DataFrame(
        {
            "id": range(n),
            "target": target,
            "x": x,
            "x_correlacionada": x * 2 + rng.normal(0, 0.05, n),
            "assimetrica": rng.exponential(2, n),
            "constante": ["sempre_igual"] * n,
            "quase_constante": (["A"] * int(n * 0.97) + ["B"] * (n - int(n * 0.97))),
        }
    )

    # outliers pontuais em x
    df.loc[df.index[:5], "x"] = 500.0

    # ausência ligada ao target (renda falta mais entre inadimplentes)
    df["renda"] = rng.normal(3000, 500, n)
    mask_inad = df["target"] == "inadimplente"
    idx_inad = df[mask_inad].sample(frac=0.6, random_state=1).index
    idx_adim = df[~mask_inad].sample(frac=0.05, random_state=1).index
    df.loc[idx_inad, "renda"] = None
    df.loc[idx_adim, "renda"] = None

    return df
