"""Simple tkinter GUI for egdol: treeview for facts/rules, query entry, results table, trace/log pane, stats panel."""
import tkinter as tk
from tkinter import ttk
from .rules_engine import RulesEngine
from .interpreter import Interpreter
from .parser import Term
import threading

class EgdolGUI:
    def __init__(self, engine: RulesEngine):
        self.engine = engine
        self.root = tk.Tk()
        self.root.title('egdol GUI')
        # capture trace lines (list of (depth, text))
        self._trace_lines = []
        self._trace_depth_filter = 999
        # trace depth var (IntVar in real tkinter)
        try:
            self._trace_depth_var = tk.IntVar()
            self._trace_depth_var.set(999)
        except Exception:
            # mock environments may not have IntVar
            class _SimpleVar:
                def __init__(self):
                    self._v = 999
                def set(self, v):
                    self._v = v
                def get(self):
                    return self._v
            self._trace_depth_var = _SimpleVar()
        self._build()

    def _build(self):
        # left: treeview for facts/rules
        left = ttk.Frame(self.root)
        left.pack(side='left', fill='y')
        self.tree = ttk.Treeview(left)
        self.tree.pack(fill='y', expand=True)
        # enable editing on double-click
        try:
            self.tree.bind('<Double-1>', self._on_tree_double)
        except Exception:
            pass
        # drag/drop bindings
        try:
            self.tree.bind('<ButtonPress-1>', self._on_tree_press)
            self.tree.bind('<B1-Motion>', self._on_tree_motion)
            self.tree.bind('<ButtonRelease-1>', self._on_tree_release)
        except Exception:
            pass
        self._dragging = None

        # center: query entry + results
        center = ttk.Frame(self.root)
        center.pack(side='left', fill='both', expand=True)
        qframe = ttk.Frame(center)
        qframe.pack(fill='x')
        try:
            ttk.Label(qframe, text='Query:').pack(side='left')
            self.query_entry = ttk.Entry(qframe)
            self.query_entry.pack(side='left', fill='x', expand=True)
            self.run_btn = ttk.Button(qframe, text='Run', command=self._on_run)
            self.run_btn.pack(side='left')
        except Exception:
            # mock ttk may not have Entry/Button/Label semantics
            self.query_entry = None
            self.run_btn = None
        self.results = ttk.Treeview(center, columns=('binding',), show='headings')
        try:
            self.results.heading('binding', text='Bindings')
        except Exception:
            pass
        self.results.pack(fill='both', expand=True)
        # allow streaming results: add a progress label
        try:
            self.stream_label = ttk.Label(center, text='')
            self.stream_label.pack(fill='x')
        except Exception:
            class _Lbl:
                def config(self, **kw):
                    pass
            self.stream_label = _Lbl()

        # right: trace and stats
        right = ttk.Frame(self.root)
        right.pack(side='right', fill='y')
        try:
            ttk.Label(right, text='Trace').pack()
            self.trace = tk.Text(right, height=10, width=40)
            self.trace.pack()
        except Exception:
            class _FallbackText:
                def __init__(self):
                    self._c = ''
                def insert(self, a, b):
                    self._c += b
                def delete(self, a, b=None):
                    self._c = ''
            self.trace = _FallbackText()

        # trace depth control
        depth_frame = ttk.Frame(right)
        depth_frame.pack(fill='x')
        try:
            ttk.Label(depth_frame, text='Max trace depth:').pack(side='left')
            self.depth_scale = ttk.Scale(depth_frame, from_=0, to=20, orient='horizontal', command=self._on_depth_change)
            self.depth_scale.set(20)
            self.depth_scale.pack(side='left', fill='x', expand=True)
        except Exception:
            self.depth_scale = type('D', (), {'get': lambda self: 999})()

        try:
            ttk.Label(right, text='Stats').pack()
            self.stats = tk.Text(right, height=6, width=40)
            self.stats.pack()
        except Exception:
            class _FallbackText2:
                def __init__(self):
                    self._c = ''
                def insert(self, a, b):
                    self._c += b
                def delete(self, a, b=None):
                    self._c = ''
            self.stats = _FallbackText2()

        self._populate_tree()
        self._refresh_stats()
        # install logging handler to capture interpreter trace messages into the trace widget
        try:
            import logging
            class _TraceHandler(logging.Handler):
                def __init__(self, gui):
                    super().__init__()
                    self.gui = gui
                def emit(self, record):
                    try:
                        msg = self.format(record)
                        # attempt to parse depth from '(depth=N)' pattern
                        depth = 0
                        import re
                        m = re.search(r'depth=(\d+)', msg)
                        if m:
                            depth = int(m.group(1))
                        self.gui._trace_lines.append((depth, msg))
                        # insert only if within current filter
                        if depth <= self.gui._trace_depth_filter:
                            self.gui.trace.insert('end', msg + '\n')
                    except Exception:
                        pass
            handler = _TraceHandler(self)
            handler.setLevel(logging.INFO)
            logging.getLogger().addHandler(handler)
        except Exception:
            pass

    def _populate_tree(self):
        # clear
        for i in self.tree.get_children():
            self.tree.delete(i)
        # group by module
        by_mod = {}
        for f in self.engine.facts:
            if ':' in f.name:
                mod, plain = f.name.split(':', 1)
            else:
                mod, plain = 'global', f.name
            by_mod.setdefault(mod, []).append((plain, f))
        for mod, items in by_mod.items():
            p = self.tree.insert('', 'end', text=mod, open=True)
            for plain, f in items:
                # store object ref in iid attribute
                iid = self.tree.insert(p, 'end', text=f'{plain}({", ".join(map(str, f.args))})')
                self.tree.set(iid, '_obj', str(id(f)))
        rnode = self.tree.insert('', 'end', text='rules', open=True)
        for r in self.engine.rules:
            iid = self.tree.insert(rnode, 'end', text=f'{r.head} :- {", ".join(map(str, r.body))}')
            self.tree.set(iid, '_obj', str(id(r)))

    def _refresh_stats(self):
        s = self.engine.stats()
        if hasattr(self, 'stats') and self.stats is not None:
            try:
                self.stats.delete('1.0', 'end')
                for k, v in s.items():
                    self.stats.insert('end', f'{k}: {v}\n')
            except Exception:
                # stats widget not fully mocked
                pass

    def _on_run(self):
        q = self.query_entry.get().strip()
        if not q:
            return
        # run in thread
        self.results.delete(*self.results.get_children())
        self.stream_label.config(text='Running...')
        threading.Thread(target=self._run_query, args=(q,), daemon=True).start()

    def _run_query(self, q: str, interp: Interpreter = None):
        # Try to parse query and use Interpreter._prove to stream results
        from .lexer import Lexer
        from .parser import Parser
        if interp is None:
            interp = Interpreter(self.engine)
        try:
            toks = Lexer(q).tokenize()
            node = Parser(toks).parse()[0]
            term = node.term if hasattr(node, 'term') else node.term if hasattr(node, 'term') else (node.term if hasattr(node, 'term') else None)
            # if it's a Query node, Parser used '?' to wrap; else if parse produced Term directly
            if hasattr(node, 'term'):
                term = node.term
            elif hasattr(node, 'head'):
                # not a query
                term = None
            else:
                term = node
        except Exception as e:
            if hasattr(self, 'trace') and self.trace is not None:
                try:
                    self.trace.insert('end', f'Parse Error: {e}\n')
                except Exception:
                    pass
            self.stream_label.config(text='')
            return
        if term is None:
            self.trace.insert('end', 'No query term parsed\n')
            self.stream_label.config(text='')
            return

        count = 0
        timer = None
        try:
            # start timeout timer if configured on interpreter
            timer = interp._start_timeout() if hasattr(interp, '_start_timeout') else None
            for s in interp._prove(term, {}):
                # check interrupt
                if getattr(interp, '_interrupt_flag', False):
                    break
                count += 1
                # format binding
                if not s:
                    try:
                        self.results.insert('', 'end', values=('false.',))
                    except Exception:
                        pass
                else:
                    parts = [f"{k} = {self._format_val(v)}" for k, v in s.items()]
                    row = '; '.join(parts)
                    try:
                        self.results.insert('', 'end', values=(row,))
                    except Exception:
                        pass
                # update stream label
                try:
                    self.stream_label.config(text=f'Got {count} result(s)')
                except Exception:
                    pass
        except Exception as e:
            try:
                self.trace.insert('end', f'Error while proving: {e}\n')
            except Exception:
                pass
        finally:
            self._refresh_stats()
            try:
                self.stream_label.config(text=f'Done ({count} result(s))')
            except Exception:
                pass
            if hasattr(interp, '_stop_timeout'):
                interp._stop_timeout(timer)

    def _update_trace_view(self):
        # refresh trace widget based on _trace_lines and _trace_depth_var
        try:
            depth = int(self._trace_depth_var.get())
        except Exception:
            try:
                depth = int(float(self.depth_scale.get()))
            except Exception:
                depth = 999
        self.trace.delete('1.0', 'end')
        for d, msg in self._trace_lines:
            if d <= depth:
                self.trace.insert('end', msg + '\n')

    def mainloop(self):
        self.root.mainloop()

    def _format_val(self, v):
        try:
            return str(v)
        except Exception:
            return repr(v)

    # Tree editing and drag/drop helpers
    def _on_tree_double(self, event):
        iid = self.tree.focus()
        if not iid:
            return
        text = self.tree.item(iid, 'text')
        # open simple dialog to edit text
        dlg = tk.Toplevel(self.root)
        dlg.title('Edit')
        ent = ttk.Entry(dlg, width=80)
        ent.insert(0, text)
        ent.pack(fill='x')
        def _save():
            new = ent.get()
            self.tree.item(iid, text=new)
            # best-effort: update engine facts/rules if we can map by object id
            try:
                objid = self.tree.set(iid, '_obj')
                if objid:
                    # find object in facts/rules by id
                    for idx, f in enumerate(self.engine.facts):
                        if str(id(f)) == objid:
                            # parse text to Term may be complex; for now update display only
                            self.engine.facts[idx] = f
                    for idx, r in enumerate(self.engine.rules):
                        if str(id(r)) == objid:
                            self.engine.rules[idx] = r
            except Exception:
                pass
            dlg.destroy()
        btn = ttk.Button(dlg, text='Save', command=_save)
        btn.pack()

    def _on_tree_press(self, event):
        iid = self.tree.identify_row(event.y)
        if iid:
            self._dragging = iid

    def _on_tree_motion(self, event):
        if not self._dragging:
            return
        # highlight potential drop target
        tgt = self.tree.identify_row(event.y)
        try:
            self.tree.selection_set(tgt)
        except Exception:
            pass

    def _on_tree_release(self, event):
        if not self._dragging:
            return
        src = self._dragging
        dst = self.tree.identify_row(event.y)
        self._dragging = None
        if not dst or dst == src:
            return
        # only allow reordering under same parent
        psrc = self.tree.parent(src)
        pdst = self.tree.parent(dst)
        if psrc != pdst:
            return
        children = list(self.tree.get_children(psrc))
        try:
            sidx = children.index(src)
            didx = children.index(dst)
        except ValueError:
            return
        # reorder in treeview
        children.pop(sidx)
        children.insert(didx, src)
        for i, ch in enumerate(children):
            self.tree.move(ch, psrc, i)
        # best-effort: reorder engine.facts or engine.rules if parent is a module or 'rules'
        try:
            parent_text = self.tree.item(psrc, 'text')
            if parent_text == 'rules':
                # rebuild engine.rules according to new order
                new_rules = []
                for ch in self.tree.get_children(psrc):
                    objid = self.tree.set(ch, '_obj')
                    for r in self.engine.rules:
                        if str(id(r)) == objid:
                            new_rules.append(r)
                            break
                if new_rules:
                    self.engine.rules = new_rules
            else:
                # assume facts under module
                new_facts = []
                for mod in self.tree.get_children(''):
                    if self.tree.item(mod, 'text') == parent_text:
                        for ch in self.tree.get_children(mod):
                            objid = self.tree.set(ch, '_obj')
                            for f in self.engine.facts:
                                if str(id(f)) == objid:
                                    new_facts.append(f)
                                    break
                if new_facts:
                    # replace facts preserving others
                    others = [f for f in self.engine.facts if (':' in f.name and f.name.split(':',1)[0] != parent_text) or (':' not in f.name and parent_text != 'global')]
                    self.engine.facts = others + new_facts
        except Exception:
            pass

    def _on_depth_change(self, v):
        try:
            self._trace_depth_filter = int(float(v))
        except Exception:
            self._trace_depth_filter = 999
        # refresh trace widget
        self.trace.delete('1.0', 'end')
        for depth, msg in self._trace_lines:
            if depth <= self._trace_depth_filter:
                self.trace.insert('end', msg + '\n')


def launch_gui(engine: RulesEngine = None):
    if engine is None:
        engine = RulesEngine()
    gui = EgdolGUI(engine)
    gui.mainloop()
