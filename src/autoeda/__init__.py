"""
Carregador de traduções do AutoEDA.

Uso típico em qualquer módulo que precise gerar texto no idioma de
saída configurado pelo usuário (AutoEDAConfig.language):

    from autoeda.i18n import get_translator
    t = get_translator(config.language)
    mensagem = t("rec.missing_high.recommendation", column="renda")

get_translator nunca levanta exceção por idioma desconhecido — cai
para pt-br (o padrão do AutoEDA) caso `lang` não seja "pt-br" nem
"en-us", já que utils.validate_language já deveria ter barrado
qualquer valor inválido antes de chegar aqui; esse fallback é só uma
salvaguarda de robustez, não a validação em si.
"""

from __future__ import annotations

from typing import Callable

from autoeda.i18n import en_us, pt_br

_CATALOGS: dict[str, dict[str, str]] = {
    "pt-br": pt_br.TRANSLATIONS,
    "en-us": en_us.TRANSLATIONS,
}

_DEFAULT_LANGUAGE = "pt-br"


def get_translator(lang: str) -> Callable[..., str]:
    """Retorna uma função `t(key, **kwargs)` que busca `key` no
    catálogo do idioma `lang` e a formata com `kwargs`
    (str.format). Levanta KeyError se a chave não existir no
    catálogo — preferível a devolver a chave crua silenciosamente,
    que esconderia strings de tradução faltando até tarde demais.
    """
    catalog = _CATALOGS.get(lang, _CATALOGS[_DEFAULT_LANGUAGE])

    def t(key: str, **kwargs: object) -> str:
        template = catalog.get(key)
        if template is None:
            raise KeyError(f"Chave de tradução '{key}' não encontrada para o idioma '{lang}'.")
        return template.format(**kwargs)

    return t


def assert_catalogs_in_sync() -> None:
    """Verifica que pt_br.TRANSLATIONS e en_us.TRANSLATIONS têm
    exatamente o mesmo conjunto de chaves.

    Chamada pelos testes (não pelo pipeline em tempo de execução) —
    um catálogo desincronizado não deveria derrubar uma análise em
    produção, mas deve ser pego antes de qualquer release.
    """
    pt_keys = set(pt_br.TRANSLATIONS.keys())
    en_keys = set(en_us.TRANSLATIONS.keys())

    only_pt = pt_keys - en_keys
    only_en = en_keys - pt_keys

    if only_pt or only_en:
        raise AssertionError(
            f"Catálogos de tradução fora de sincronia. "
            f"Só em pt-br: {sorted(only_pt)}. Só em en-us: {sorted(only_en)}."
        )
