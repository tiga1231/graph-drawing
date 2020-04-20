from models.stress import Stress

from random import random
import networkx as nx

from flask import Flask, request, url_for, render_template
from flask_socketio import SocketIO, emit
from flask.json import jsonify

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sdddvldffnsnd0dn2n3enfn49n429nf39fn3nf94fsk'
socketio = SocketIO(app)

position_optimizer = None

def graph2json(G):
    return {
        'nodes': list(G.nodes),
        'edges': list(G.edges)
    }



@app.route('/')
def index():
    return render_template('index.html')
    

# @app.route('/post', methods=['GET', 'POST'])
# def post():
#     if request.method == 'GET':
#         return 'OK'
#     else:
#         return jsonify(stress.get())


@socketio.on('graph')
def graph_info(req):
    global position_optimizer
    print('requesting graph:\n', req)
    response = {'msg':{}}
    if 'graph' in req['items']:
        if req['meta']['type'] == 'balanced_tree':
            r = req['meta']['branches']
            h = req['meta']['height']
            graph = nx.balanced_tree(r,h)
            position_optimizer = Stress(graph)

            graph = graph2json(graph)
            response['graph'] = graph
            response['msg']['graph'] = 'available'

    if 'pos' in req['items']:
        steps = 100
        newPos, diff = position_optimizer.update(steps)
        if diff > 1e-4:
            newPos = newPos.detach().cpu().numpy().tolist()
            response['pos'] = newPos
            response['msg']['pos'] = 'available'
        else:
            response['msg']['pos'] = 'unchanged'

    emit('graph', response)



if __name__ == '__main__':
    print('start...')
    socketio.run(app, port=10001, host='0.0.0.0', debug=True)
