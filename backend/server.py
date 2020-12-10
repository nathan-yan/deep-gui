# server that takes json post requests and returns the network it creates
from flask import Flask, request
import compiler

import json

app = Flask(__name__)


@app.route('/compile', methods=["GET", "POST"])
def test():
    if request.method == "POST":
        networkJson = request.get_json(force = True)
        print(networkJson)

        compiledNetwork = compiler.write('blank.py', networkJson)

        with open("compiled.py", 'w') as compiled:
            compiled.write(compiledNetwork)
        
        return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 


if __name__ == "__main__":
    app.run(debug=True)
