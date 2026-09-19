"""
Exportação do JSON de recomendações do AutoEDA.

Este módulo é intencionalmente simples: recommendations.py já produz
a estrutura completa e serializável ({"schema_version": ...,
"recommendations": [...]}); aqui só cuidamos de escrever esse dict em
disco (ou devolvê-lo como string), sem reprocessar nada.

Mantido separado de recommendations.py porque "gerar as recomendações"
(lógica de negócio) e "salvar/serializar em um formato de arquivo"
(E/S) são responsabilidades diferentes — útil, por exemplo, se no
futuro o AutoEDA precisar exportar o mesmo conteúdo em outro formato
(YAML, por exemplo) sem tocar em recommendations.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def recommendations_to_json_string(recommendations_result: dict[str, Any], indent: int = 2) -> str:
    """Serializa o resultado de
    recommendations.generate_recommendations para uma string JSON.

    `ensure_ascii=False` preserva acentuação em português no arquivo
    de saída, em vez de escapar cada caractere acentuado como \\uXXXX
    — mais legível para quem abrir o JSON diretamente.
    """
    return json.dumps(recommendations_result, indent=indent, ensure_ascii=False)


def export_recommendations_json(
    recommendations_result: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Escreve o resultado de
    recommendations.generate_recommendations em um arquivo .json.

    Cria os diretórios intermediários de `output_path` se necessário,
    para o chamador não precisar garantir isso separadamente. Retorna
    o Path final (resolvido), útil para o core.py reportar ao usuário
    onde o arquivo foi salvo.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(recommendations_to_json_string(recommendations_result), encoding="utf-8")

    return path.resolve()
