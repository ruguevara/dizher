# -*- coding: utf-8 -*-
import sys
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    multiprocessing.set_start_method('fork')
    from .app import DizherApp
    app = DizherApp()
    initial_fname = sys.argv[1] if len(sys.argv) > 1 else None
    app.open_image(initial_fname)
    app.event_loop()
