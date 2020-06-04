// main
let isPlaying = true;
window.onload = function(){
  let graphTypeSelect = d3.select('#graphType');
  let fn = `data/${graphTypeSelect.node().value}.json`;

  let lrSlider = d3.select('#lr');
  window.lr0 = +Math.exp(lrSlider.node().value);
  window.lr = +Math.exp(lrSlider.node().value);
  window.lrSlider = lrSlider;
  let lrText = d3.select('#lrText')
  .text(lr.toFixed(4));

  let momentumSlider = d3.select('#momentum');
  window.momentum = +momentumSlider.node().value;
  let momentumText = d3.select('#momentumText')
  .text(momentum);

  let stressSlider = d3.select('#stressSlider');
  let stressLabel = d3.select('#stressLabel');
  let crossingAngleSlider = d3.select('#crossingAngleSlider');
  let crossingAngleLabel = d3.select('#crossingAngleLabel');
  let angularResolutionSlider = d3.select('#angularResolutionSlider');
  let angularResolutionLabel = d3.select('#angularResolutionLabel');
  let neighborSlider = d3.select('#neighborSlider');
  let neighborLabel = d3.select('#neighborLabel');
  let edgeUniformitySlider = d3.select('#edgeUniformitySlider');
  let edgeUniformityLabel = d3.select('#edgeUniformityLabel');
  let crossingNumberSlider = d3.select('#crossingNumberSlider');
  let crossingNumberLabel = d3.select('#crossingNumberLabel');

  let vertexResolutionSlider = d3.select('#vertexResolutionSlider');
  let vertexResolutionLabel = d3.select('#vertexResolutionLabel');

  let gabrielSlider = d3.select('#gabrielSlider');
  let gabrielLabel = d3.select('#gabrielLabel');
  let aspectRatioSlider = d3.select('#aspectRatioSlider');
  let aspectRatioLabel = d3.select('#aspectRatioLabel');

  // let areaSlider = d3.select('#areaSlider');
  // let areaLabel = d3.select('#areaLabel');
  // let desiredAreaSlider = d3.select('#desiredAreaSlider');
  // let desiredAreaLabel = d3.select('#desiredAreaLabel');


  let coef = {
    stress: +stressSlider.node().value,
    crossing_angle: +crossingAngleSlider.node().value,
    neighbor: +neighborSlider.node().value,
    edge_uniformity: +edgeUniformitySlider.node().value,
    crossing_number: +crossingNumberSlider.node().value,
    angular_resolution: +angularResolutionSlider.node().value,
    vertex_resolution: +vertexResolutionSlider.node().value,
    gabriel: +gabrielSlider.node().value,
    aspect_ratio: +aspectRatioSlider.node().value,
    // area: (areaSlider.node() !== null ? +areaSlider.node().value : 0.0),
  };

  let sampleSize = 5;
 

  let maxPlotIter = 50;
  let niter = 20000;
  let maxMetricSize = 10;

  let optimizers = [tf.train.momentum(lr, momentum, false)];
  window.optimizers = optimizers;
  let losses = [];
  let metrics = [];
  window.metrics = metrics;
  let animId = 0;

  let svg_loss = d3.select('#loss');
  let svg_graph = d3.select('#graph');

  let svg_metrics = d3.selectAll('.metric');

  let playButton = d3.select('#play');
  window.playButton = playButton;
  let resetButton = d3.select('#reset');

  updateSvgSize(svg_loss, svg_graph);
  window.addEventListener('resize', ()=>{updateSvgSize(svg_loss, svg_graph)});

  function loadGraph(fn){
    d3.json(fn).then((graph)=>{
      window.graph = graph;
      let x;
      function reset(){
        console.log('reset');
        x = tf.variable(
          tf.randomUniform([graph.nodes.length,2], -1, 1)
        );
        boundaries = undefined;
        historicalWorst = {};
        if(lrSlider.on('input')){
          lrSlider.on('input')(lr0);
        }

      }

      if(graph.initPositions){
        x = tf.variable(tf.tensor2d(graph.initPositions));
      }else{
        reset();
        optimizers[0] = tf.train.momentum(lr, momentum, false);
      }

      preprocess(graph, x.arraySync());

      let n_neighbors = graph.graphDistance
      .map((row)=>{
        return row.reduce((a,b)=>b==1?a+1:a, 0);
      });
      let adj = graph.graphDistance.map(row=>row.map(d=>d==1.0 ? 1.0 : 0.0));
      adj = tf.tensor2d(adj);
      let graphDistance = tf.tensor2d(graph.graphDistance);
      let stressWeight = tf.tensor2d(graph.weight);
      let edgePairs = getEdgePairs(graph);
      let neighbors = graph2neighbors(graph);
      let edges = graph.edges.map(d=>[d.source.index, d.target.index]);

      let dataObj = {
        sampleSize,
        x, 
        graphDistance, 
        adj,
        stressWeight,
        graph,
        animId,
        coef,
        n_neighbors,
        edgePairs,
        neighbors,
        edges,
        // desiredArea: desiredAreaSlider.node()===null? 0.0: +desiredAreaSlider.node().value,
      };
      window.dataObj = dataObj;

      function play(){
        train(dataObj, niter, optimizers, (record)=>{
          // console.log(record);
          metrics.push(record.metrics);
          losses.push(record.loss);
          if (losses.length>maxPlotIter){
            losses = losses.slice(losses.length-maxPlotIter);
          }
          if (metrics.length > maxMetricSize){
            metrics = metrics.slice(metrics.length-maxMetricSize);
          }
          traceLoss(svg_loss, losses, maxPlotIter);
          traceMetrics(svg_metrics, metrics, maxMetricSize);
          window.metrics = metrics;
          if (losses.length >= 10){
            let n = losses.length;
            let firstSlice = losses.slice(Math.floor(n/2), Math.floor(n/4*3));
            let secondSlice = losses.slice(Math.floor(n/4*3), n);
            let avgLoss0 = math.mean(firstSlice);
            let avgLoss1 = math.mean(secondSlice);
            if(avgLoss1 > avgLoss0){
              lrSlider.on('input')(Math.max(lr/1.01, 0.001 ));
            }
          }
          updateNodePosition(graph, x.arraySync());
          drawGraph(svg_graph, graph);
        });
      }
      play();
      
      //interactions
      playButton.on('click', function(shouldPlay){
        if(shouldPlay === undefined){
          isPlaying = !isPlaying;
        }else{
          isPlaying = shouldPlay;
        }
        if(isPlaying){
          playButton.attr('class', 'fas fa-pause-circle');
          play();
        }else{
          playButton.attr('class', 'fas fa-play-circle');
        }
      });

      resetButton.on('click', function(){
        reset();
        svg_graph.xDomain = undefined;
        let xy = x.arraySync();
        graph.nodes.forEach((d,i)=>{
          d.x = xy[i][0];
          d.x = xy[i][1];
        });
        dataObj.x = x;
        drawGraph(svg_graph, graph);
      });

      graphTypeSelect.on('change', function(){
        let fn = d3.select(this).node().value;
        fn = `data/${fn}.json`;
        cancelAnimationFrame(dataObj.animId);
        loadGraph(fn);
      });

      lrSlider.on('input', function(value){
        window.lr = value || Math.exp(+d3.select(this).node().value);
        lrText.text(lr.toFixed(4));
        optimizers[0] = tf.train.momentum(lr, momentum, false);
        lrSlider.property('value', Math.log(lr));
      });

      momentumSlider.on('input', function(){
        window.momentum = +d3.select(this).node().value;
        momentumText.text(window.momentum);
        optimizers[0] = tf.train.momentum(window.lr, momentum, false);
      });

      stressSlider.on('input', function(){
        let value = +stressSlider.node().value;
        stressLabel.text(value.toFixed(2));
        coef['stress'] = value;
      });
      stressSlider.on('input')();

      crossingAngleSlider.on('input', function(){
        let value = +crossingAngleSlider.node().value;
        crossingAngleLabel.text(value.toFixed(2));
        coef['crossing_angle'] = value;
      });
      crossingAngleSlider.on('input')();

      neighborSlider.on('input', function(){
        let value = +neighborSlider.node().value;
        neighborLabel.text(value.toFixed(2));
        coef['neighbor'] = value;
      });
      neighborSlider.on('input')();

      edgeUniformitySlider.on('input', function(){
        let value = +edgeUniformitySlider.node().value;
        edgeUniformityLabel.text(value.toFixed(2));
        coef['edge_uniformity'] = value;
      });
      edgeUniformitySlider.on('input')();

      crossingNumberSlider.on('input', function(){
        let value = +crossingNumberSlider.node().value;
        crossingNumberLabel.text(value.toFixed(2));
        coef['crossing_number'] = value;
      });
      crossingNumberSlider.on('input')();

      angularResolutionSlider.on('input', function(){
        let value = +angularResolutionSlider.node().value;
        angularResolutionLabel.text(value.toFixed(2));
        coef['angular_resolution'] = value;
      });
      angularResolutionSlider.on('input')();

      // areaSlider.on('input', function(){
      //   let value = +areaSlider.node().value;
      //   areaLabel.text(value.toFixed(2));
      //   coef['area'] = value;
      // });
      // areaSlider.on('input')();
      // desiredAreaSlider.on('input', function(){
      //   let value = +desiredAreaSlider.node().value;
      //   desiredAreaLabel.text(value.toFixed(2));
      //   dataObj['desiredArea'] = value;
      // });
      // desiredAreaSlider.on('input')();


      vertexResolutionSlider.on('input', function(){
        let value = +vertexResolutionSlider.node().value;
        vertexResolutionLabel.text(value.toFixed(2));
        coef['vertex_resolution'] = value;
      });
      vertexResolutionSlider.on('input')();

      gabrielSlider.on('input', function(){
        let value = +gabrielSlider.node().value;
        gabrielLabel.text(value.toFixed(2));
        coef['gabriel'] = value;
      });
      gabrielSlider.on('input')();

      aspectRatioSlider.on('input', function(){
        let value = +aspectRatioSlider.node().value;
        aspectRatioLabel.text(value.toFixed(2));
        coef['aspect_ratio'] = value;
      });
      aspectRatioSlider.on('input')();



    });
  }//loadGraph end
  loadGraph(fn);
};//onload end

