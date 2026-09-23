# -*- coding: utf-8 -*-
import signal
import sys
import multiprocessing


def main():
    # NOTE: do not remove or rename this function, it's called as a console script:
    #       `dizher = dizher.gui.__main__:main`
    multiprocessing.freeze_support()
    multiprocessing.set_start_method('fork')
    from .app import DizherApp
    app = DizherApp()
    initial_fname = sys.argv[1] if len(sys.argv) > 1 else None
    # Tk swallows KeyboardInterrupt inside its callbacks: turn Ctrl-C into a regular Exit event instead
    signal.signal(signal.SIGINT, lambda *_: app.window.write_event_value('Exit', None))
    app.open_image(initial_fname)
    try:
        app.event_loop()
    finally:
        app.worker.abort()


if __name__ == "__main__":
    main()
