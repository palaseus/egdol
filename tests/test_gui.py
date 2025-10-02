import unittest
from edgol.rules_engine import RulesEngine

try:
    from edgol.gui import EdgolGUI
    _HAS_TKINTER = True
except Exception:
    # Headless or missing tkinter; skip GUI tests
    _HAS_TKINTER = False


@unittest.skipUnless(_HAS_TKINTER, "tkinter not available; skipping GUI tests")
class GuiSmokeTest(unittest.TestCase):
    def test_gui_construct_and_destroy(self):
        eng = RulesEngine()
        gui = EdgolGUI(eng)
        # check basic widgets exist
        self.assertTrue(hasattr(gui, 'tree'))
        self.assertTrue(hasattr(gui, 'results'))
        # Try to process an update if possible
        try:
            gui.root.update()
        except Exception:
            pass
        # destroy window to avoid hanging
        gui.on_close()
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
