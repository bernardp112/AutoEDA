"""Testes de autoeda.i18n."""

from __future__ import annotations

import pytest

from autoeda.i18n import assert_catalogs_in_sync, get_translator


class TestCatalogsInSync:
    def test_pt_br_and_en_us_have_same_keys(self):
        assert_catalogs_in_sync()  # nao deve levantar excecao


class TestGetTranslator:
    def test_pt_br_translation(self):
        t = get_translator("pt-br")
        assert t("rec.constant.recommendation", column="renda") == "Remover a coluna 'renda' do dataset."

    def test_en_us_translation(self):
        t = get_translator("en-us")
        assert t("rec.constant.recommendation", column="renda") == "Remove the 'renda' column from the dataset."

    def test_unknown_language_falls_back_to_pt_br(self):
        t = get_translator("fr-fr")
        assert t("report.table.metric") == "Métrica"

    def test_unknown_key_raises(self):
        t = get_translator("pt-br")
        with pytest.raises(KeyError):
            t("chave.que.nao.existe")
