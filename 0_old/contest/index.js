//main
let graphTypeSelect = d3.select('#graphType');
let fn = `data/${graphTypeSelect.node().value}.json`;

let svg_graph = d3.select('#graph');
updateSvgSize(svg_graph);

let graph, simulation;
d3.json(fn).then((graph0)=>{
  graph = graph0;
  initNodePosition(graph);
  simulation = startSimulation(graph, svg_graph);
  drawGraph(graph, svg_graph);
});


let playButton = d3.select('#play');
let isPlaying = true;
playButton.on('click', function(shouldPlay){
  isPlaying = !isPlaying;
  if(isPlaying){
    playButton.attr('class', 'fas fa-pause-circle');
    simulation.restart();
  }else{
    simulation.stop();
    playButton.attr('class', 'fas fa-play-circle');
  }
});


let boostButton = d3.select('#boost');
boostButton.on('click', ()=>{
  simulation.alpha(1.0);
  simulation.restart();
});


let resetButton = d3.select('#reset');
resetButton.on('click', ()=>{
  //todo: reset graph pos
  initNodePosition(graph);
  simulation.alpha(1.0);
  simulation.restart();
});


