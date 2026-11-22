"""Adaptador de entrada da interface Streamlit.

A implementacao visual permanece em ``dockingpp.gui`` para compatibilidade;
este modulo fornece o ponto de entrada arquitetural para a aplicacao.
"""

from dockingpp.gui.app import main

__all__ = ["main"]


if __name__ == "__main__":
    main()

