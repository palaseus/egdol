import unittest
from unittest import mock

# Mock tkinter elements used by egdol.gui
class FakeTk:
    def __init__(self):
        self._widgets = []
    def title(self, t):
        pass
    def mainloop(self):
        pass
    def destroy(self):
        pass
    def withdraw(self):
        pass

class FakeTreeview:
    def __init__(self, *args, **kwargs):
        self._items = {}
        self._next = 1
    def insert(self, parent, index, iid=None, text=None, values=None, **kwargs):
        # accept arbitrary kwargs for compatibility (e.g., open=True)
        iid = iid or str(self._next)
        self._next += 1
        self._items[iid] = {'text': text, 'values': values}
        # store any extra kwargs
        if kwargs:
            self._items[iid].update(kwargs)
        return iid
    def item(self, iid, **kwargs):
        if kwargs:
            self._items[iid].update(kwargs)
        return self._items.get(iid)
    def bind(self, evt, handler):
        # store binding for invocation
        setattr(self, f'_bind_{evt}', handler)
    def move(self, iid, parent, index):
        # simulate move by keeping dict order
        return
    def get_children(self):
        return list(self._items.keys())
    def delete(self, *iids):
        for iid in iids:
            if iid in self._items:
                del self._items[iid]
    def set(self, iid, key, val=None):
        # support both get and set semantics
        if val is None:
            return self._items.get(iid, {}).get(key)
        self._items.setdefault(iid, {})[key] = val
    def parent(self, iid):
        # not tracking parent hierarchy in mock; return root
        return ''
    def identify_row(self, y):
        # return first child if exists
        ch = self.get_children()
        return ch[0] if ch else ''
    def selection_set(self, iid):
        self._selected = iid
    def focus(self):
        return getattr(self, '_selected', None)
    def pack(self, *args, **kwargs):
        return

class FakeText:
    def __init__(self):
        self._content = ''
    def insert(self, idx, txt):
        self._content += txt
    def delete(self, a, b=None):
        self._content = ''
    def get(self, a='1.0', b='end'):
        return self._content
    def pack(self, *a, **k):
        return

class GuiMockTests(unittest.TestCase):
    def test_gui_edit_and_stream_and_trace(self):
        # Patch tkinter and ttk in the egdol.gui module import path
        import sys
        fake_tk_mod = mock.MagicMock()
        fake_tk_mod.Tk.return_value = FakeTk()
        fake_ttk = mock.MagicMock()
        fake_ttk.Treeview = FakeTreeview
        fake_tk_mod.ttk = fake_ttk
        fake_tk_mod.Text = FakeText
        fake_tk_mod.IntVar = lambda: type('V', (), {'_v': 999, 'set': lambda self, v: setattr(self, '_v', v), 'get': lambda self: getattr(self, '_v')})()
        sys.modules['tkinter'] = fake_tk_mod
        sys.modules['tkinter.ttk'] = fake_ttk
        # import gui after injecting mocks
        from egdol.gui import EgdolGUI
        from egdol.rules_engine import RulesEngine
        from egdol.interpreter import Interpreter
        eng = RulesEngine()
        interp = Interpreter(eng)
        gui = EgdolGUI(eng)
        # simulate inserting a fact via tree edit
        tree = gui.tree
        iid = tree.insert('', 'end', text='fact1', values=('fact',))
        tree.item(iid, text='fact1_edited')
        self.assertIn(iid, tree.get_children())
        # simulate streaming results: patch interpreter._prove to yield two results
        def fake_prove(goal, subst, depth=0):
            yield {'X': 1}
            yield {'X': 2}
        interp._prove = fake_prove
        # run a query via GUI streaming method
        gui._run_query('? p(X).', interp)
        # check results appended to results view
        # results widget is a Treeview; ensure it has children
        res_children = gui.results.get_children()
        self.assertTrue(len(res_children) >= 2)
        # simulate trace folding: set trace lines and call fold
        gui._trace_lines = [(0, 'enter p/1'), (1, 'enter q/1'), (2, 'enter r/1')]
        gui._update_trace_view()
        # simulate depth toggle
        gui._trace_depth_var.set(1)
        gui._update_trace_view()

if __name__ == '__main__':
    unittest.main()
