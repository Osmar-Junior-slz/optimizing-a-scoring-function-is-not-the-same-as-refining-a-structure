from __future__ import annotations

from pathlib import Path
import sys

# `streamlit run path/to/app.py` adds the UI folder to sys.path, not src.
# Prefer this checkout over a global editable install from another project.
_source = Path(__file__).resolve().parents[2]
if str(_source) in sys.path:
    sys.path.remove(str(_source))
sys.path.insert(0, str(_source))
for _package_name in ("docking_confidence", "dockingpp"):
    _loaded_package = sys.modules.get(_package_name)
    if _loaded_package is not None and Path(_loaded_package.__file__).resolve().parent != _source / _package_name:
        raise RuntimeError(
            f"O processo carregou {_package_name} de outro projeto. "
            "Encerre a interface com Ctrl+C e abra iniciar_sistema.cmd nesta pasta."
        )

import streamlit as st

from docking_confidence.ui.calibrator_page import render_calibrator_page
from docking_confidence.ui.decision_page import render_decision_page


st.set_page_config(page_title="Generalized Venn–Abers 2025", page_icon=":material/model_training:", layout="wide")
with st.expander("Revisão do artigo de docking"):
    st.write("Na pasta do projeto, execute a etapa desejada no PowerShell. Cada comando usa as entradas arquivadas dos 50 complexos.")
    st.code(".\\iniciar_sistema.cmd revisao curvas\n"
            ".\\iniciar_sistema.cmd revisao controle\n"
            ".\\iniciar_sistema.cmd revisao taxa-poses\n"
            ".\\iniciar_sistema.cmd revisao linhagem", language="powershell")
    st.caption("Curvas analisa os registros das 600 buscas. Controle sorteia as poses e avalia a vencedora de cada complexo e região. "
               "Após o controle, taxa-poses mede a frequência de DockQ ≥ 0,23 entre todas as poses sorteadas. "
               "Linhagem repete as 300 buscas semeadas. Os resultados ficam em output/revisao_professor.")
    chart = _source.parent / "output/revisao_professor/convergencia/convergence.png"
    if chart.is_file():
        st.image(str(chart), caption="Mediana do melhor valor observado da função objetivo, por número de avaliações.")
st.title("Criar Generalized Venn–Abers (2025)")
render_calibrator_page()

with st.expander("Validação da decisão e teste separado", expanded=False):
    render_decision_page()
