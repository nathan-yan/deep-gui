# server that takes json post requests and returns the network it creates
from flask import Flask, request
import compiler

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def test():
    if request.method == "POST":
        networkJson = request.json
        compiledNetwork = compiler.write(compiler.compile(networkJson)).replace('\n', '<br>').replace(' ', '&nbsp')
        return compiledNetwork

if __name__ == "__main__":
    app.run(debug=True)