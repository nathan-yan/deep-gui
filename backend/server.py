# server that takes json post requests and returns the network it creates
from flask import Flask, request
import compiler

import json

app = Flask(__name__)
cached_network = {'network' : None}

import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
class MyHandler(FileSystemEventHandler):
    def on_modified(self, event):
        print("Got it!", event.src_path)

        if event.src_path == '../example_workspace\\template.py':
            # rewrite
            write()

event_handler = MyHandler()
observer = Observer()


def write():
    compiledNetwork = compiler.write('../example_workspace/template.py', cached_network['network'])

    with open("../example_workspace/compiled.py", 'w') as compiled:
        compiled.write(compiledNetwork)

@app.route('/compile', methods=["GET", "POST"])
def test():
    if request.method == "POST":
        networkJson = request.get_json(force = True)
        print(networkJson)

        # replace the cached network
        cached_network['network'] = networkJson

        write()
        
        return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 


if __name__ == "__main__":
    observer.schedule(event_handler, path='../example_workspace', recursive=False)
    observer.start()

    try:
        app.run(debug=True)

    finally:
        observer.stop()
        observer.join()
