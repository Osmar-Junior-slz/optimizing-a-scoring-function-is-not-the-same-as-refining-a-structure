"""GUI page registry."""

from dockingpp.gui.pages.calibration import CalibrationPage
from dockingpp.gui.pages.config import ConfigPage
from dockingpp.gui.pages.dashboard import ScientificDashboardPage
from dockingpp.gui.pages.docking import DockingPage
from dockingpp.gui.pages.home import HomePage
from dockingpp.gui.pages.pdb_prep import PdbPrepPage
from dockingpp.gui.pages.reports import ReportsPage

__all__ = [
    "CalibrationPage",
    "ConfigPage",
    "ScientificDashboardPage",
    "DockingPage",
    "HomePage",
    "PdbPrepPage",
    "ReportsPage",
]
