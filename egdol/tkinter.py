"""Minimal headless tkinter shim used for tests when real tkinter is unavailable.

This provides the small subset of the tkinter/ttk API that edgol.gui relies on.
It's intentionally simple and non-graphical; methods are no-ops or simple state holders
so GUI unit tests can exercise logic in headless CI.
"""
class IntVar:
    def __init__(self):
        self._v = 0
    def set(self, v):
        self._v = v
    def get(self):
        return self._v


class Tk:
    def __init__(self):
        self._title = ''
    def title(self, t):
        self._title = t
    def mainloop(self):
        return
    def update(self):
        return
    def destroy(self):
        return
    def withdraw(self):
        return


class Toplevel(Tk):
    pass


class Text:
    def __init__(self, master=None, height=None, width=None):
        self._c = ''
    def insert(self, pos, txt):
        self._c += txt
    def delete(self, a, b=None):
        self._c = ''
    def pack(self, **kw):
        return


class _WidgetBase:
    def pack(self, **kw):
        return
    def bind(self, *a, **k):
        return


class ttk:
    class Frame(_WidgetBase):
        def __init__(self, master=None):
            self._children = []
    class Label(_WidgetBase):
        def __init__(self, master=None, text=''):
            self._text = text
    class Entry(_WidgetBase):
        def __init__(self, master=None):
            self._txt = ''
        def insert(self, idx, txt):
            self._txt = str(txt)
        def get(self):
            return self._txt
        def pack(self, **kw):
            return
    class Button(_WidgetBase):
        def __init__(self, master=None, text='', command=None):
            self._cmd = command
        def pack(self, **kw):
            return
    class Treeview(_WidgetBase):
        def __init__(self, master=None, columns=None, show=None):
            self._items = {}
            self._order = []
            self._parent = {}
            self._values = {}
            self._counter = 0
        def insert(self, parent, index, text=''):
            iid = f'i{self._counter}'
            self._counter += 1
            self._items[iid] = text
            self._order.append(iid)
            self._parent[iid] = parent
            return iid
        def get_children(self, iid=''):
            return [k for k, p in self._parent.items() if p == iid]
        def delete(self, iid):
            if iid in self._items:
                del self._items[iid]
        def set(self, iid, key, val):
            self._values[iid] = val
        def item(self, iid, opt, **kw):
            if opt == 'text':
                return self._items.get(iid, '')
            return self._items.get(iid, '')
        def focus(self):
            return self._order[0] if self._order else ''
        def identify_row(self, y):
            return self._order[0] if self._order else ''
        def selection_set(self, iid):
            return
        def move(self, iid, parent, index):
            return
        def heading(self, col, text=''):
            return
        def pack(self, **kw):
            return
    class Scale(_WidgetBase):
        def __init__(self, master=None, from_=0, to=1, orient=None, command=None):
            self._v = from_
        def set(self, v):
            self._v = v
        def get(self):
            return self._v


# expose commonly used names
Tk = Tk
Text = Text
Toplevel = Toplevel
IntVar = IntVar
ttk = ttk
