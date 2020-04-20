let server = document.location.href;
let socket = io();

let graphMeta = {
    type: 'balanced_tree',
    branches: 2,
    height: 6,
};

let initReq = {
  'meta': graphMeta,
  'items': ['graph', 'pos'],
};

let posReq = {
  'meta': graphMeta,
  'items': ['pos'],
}


let graph = null;

let svg = d3.select('#graph-view')
  .attr('width', window.innerWidth)
  .attr('height',window.innerHeight);


socket.on('connect', function() {
  socket.emit('graph', initReq);
  //   socket.emit('graph_positions', {graph_id: 'tree'})

});

socket.on('graph', function(response){
  console.log('EVENT graph');
  let data = {};
  if('graph' in response){
    data.graph = utils.graph2data(response.graph);
  }
  if('pos' in response && response.msg.pos == 'available'){
    data.pos = response.pos;
  }
  initGraph(svg, data);
  setTimeout(()=>{socket.emit('graph', posReq);}, 1000);

//   // socket.emit('node_positions', {graph_id: graph_id})
});


// socket.on('graph_positions', function(response){
//   console.log('response', response);
//   if(response.msg == 'unchanged'){
//     console.log(response.graph_id, 'node positions unchanged');
//   }else if(response.msg == 'available'){
//     console.log(response.graph_id, 'node positions available');
//     console.log(response.pos);
//     initGraph(svg, response.pos);
//   }
// });




