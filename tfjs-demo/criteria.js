function preprocess(graph, initPos){
  graph.scalingFactor = 1.0;
  graph.snapToInt = false;
  console.log(`[w,h] = [${graph.width},${graph.height}]`);
  if(!graph.hasOwnProperty('width')){
    graph.width = 1e6;
  }
  if(!graph.hasOwnProperty('height')){
    graph.height = 1e6;
  }
  graph.center = [graph.width/2||0, graph.height/2||0];

  graph.nodes.forEach((d,i)=>{
    if(!d.hasOwnProperty('index')){
      d.index = i;
    }
  });
  
  if (initPos !== undefined){
    graph.nodes.forEach((d,i)=>{
      d.x = initPos[d.index][0];
      d.y = initPos[d.index][1];
    });
  }

  let id2node = {};
  graph.nodes.forEach((d,i)=>{
    id2node[d.id] = d;
  });

  graph.edges.forEach((d,i)=>{
    d.source = id2node[d.source];
    d.target = id2node[d.target];
  });
  return graph;
}

function postprocess(x, graph){
  graph.xmin = d3.min(x, d=>d[0]);
  graph.ymin = d3.min(x, d=>d[1]);

  let y = x.map(d=>{
    d = d.slice();
    d[0] = (d[0] - graph.xmin) * graph.scalingFactor;
    d[1] = (d[1] - graph.ymin) * graph.scalingFactor;
    if(graph.snapToInt){
      d[0] = Math.round(d[0]);
      d[1] = Math.round(d[1]);
    }
    return d;
  });
  return y;
}




function randInt(a,b){
  return parseInt(Math.random() * (b-a) + a);
}


function pairwise_distance(x){
  return tf.tidy(()=>{
    let xNormSquared = x.norm('euclidean', 1, true).pow(2);
    let pairwiseDotProduct = x.matMul(x.transpose());
    let pdistSqaured = pairwiseDotProduct.mul(-2)
    .add(xNormSquared)
    .add(xNormSquared.transpose());
    let pdist = pdistSqaured.clipByValue(0.0, Infinity).sqrt();
    return pdist;
  });
}


function stress_loss(pdist, graphDistance, weight){
  return tf.tidy(()=>{
    let n = pdist.shape[0];
    let mask = tf.scalar(1.0).sub(tf.eye(n));
    let numerator = graphDistance.mul(weight).mul(pdist).mul(mask).sum();
    let denominator = graphDistance.pow(2).mul(weight).mul(mask).sum();
    let optimalScaling = numerator.div( denominator );

    let stress = pdist.sub(graphDistance).square().mul(weight).sum().div(2);
    let loss = stress.div(2);

    //metric
    let pdist_normalized = pdist.div(optimalScaling);
    let metric = pdist_normalized.sub(graphDistance).square().mul(weight).sum().div(2);
    metric = metric.dataSync()[0];
    // return [loss, loss.dataSync()[0], pdist];
    return [loss, metric, pdist_normalized.arraySync()];
  });
}

function stress_loss_2(pdist, graphDistance, weight){
  let n = pdist.shape[0];
  let upperTriangularIndices = d3.range(n*n).filter(i=>{
    let col = i % n;
    let row = Math.floor(i / n);
    return col > row;
  });

  return tf.tidy(()=>{
    pdist = pdist.flatten().gather(upperTriangularIndices);
    graphDistance = graphDistance.flatten().gather(upperTriangularIndices);
    weight = weight.flatten().gather(upperTriangularIndices);

    let stress = pdist.sub(graphDistance).square().mul(weight).sum().div(2);
    let loss = stress.div(2);
    return [loss, 0, 0];
  });
}


function edge_uniformity_loss(pdist, adj){
  let n_nodes = pdist.shape[0];
  let N = adj.sum(); // = number of edges *2
  return tf.tidy(()=>{
    let mean = pdist.mul(adj).sum().div(N);

    // let mean_data = mean.dataSync()[0];
    // if(mean_data < 0.5){
    //   mean = mean.add(0.5-mean_data); //for stability
    // }

    let loss = pdist
    .sub(mean.add(0.1))
    .mul(adj)
    .square()
    .sum()
    // .div(N)
    .div(10)
    .sqrt();

    let metric = pdist
    .sub(mean)
    .div(mean)
    .mul(adj)
    .square()
    .sum()
    .div(N)
    .sqrt();

    return [loss, metric.dataSync()[0]];
  });
}


function vertex_resolution_loss(pdist, resolution){
  let n = pdist.shape[0];
  resolution = resolution || 1.0/Math.sqrt(n);

  return tf.tidy(()=>{
    let mask = tf.scalar(1.0).sub(tf.eye(pdist.shape[0]));
    
    let max = pdist.max();
    let pdist_normalized = pdist.div(max.add(0.001));

    // let r = pdist_normalized.sub(resolution).div(-resolution).relu();
    // let loss = r.pow(2).mul( tf.scalar(3).sub(r.mul(2)) );//smoothstep: r^2 * (3-2r)

    let r = pdist_normalized.sub(resolution).div(-resolution).relu();
    // let loss = tf.scalar(1.0).sub(tf.scalar(1.001).sub(r.pow(2)));
    let loss = r.pow(2);
    loss = loss.mul(mask).sum().mul(0.02);

    let metric = 1.0 - r.mul(mask).max().dataSync()[0];
    
    return [loss, metric];
  });
}

function angular_resolution_metric(x, neighbors){
  x = x.arraySync();
  let n = x.length;
  let metric_min = Infinity;
  for(let i=0; i<n; i++){
    let center_node = x[i];
    let neighbor_nodes = neighbors[i].map(k=>x[k]);
    let v = neighbor_nodes.map(d=>numeric.sub(d, center_node));

    // https://math.stackexchange.com/questions/1327253/how-do-we-find-out-angle-from-x-y-coordinates
    let angles = v.map(d=>{
      let [x,y] = d;
      return 2*Math.atan(y/(x+Math.sqrt(x*x+y*y))) + Math.PI;
    });
    angles.sort();
    let ang1 = angles.slice(1);
    let ang0 = angles.slice(0, angles.length-1);
    let ang_diff = numeric.sub(ang1, ang0);
    let last_diff = Math.PI*2 - (angles[angles.length-1] - angles[0]);
    ang_diff.push(last_diff);

    let target = Math.PI*2 / neighbor_nodes.length;
    let metric_i = math.min(ang_diff) / target;
    if(metric_i < metric_min){
      metric_min = metric_i;
    }
  }
  let metric = metric_min;
  return metric;
}


function std(x){
  return tf.tidy(()=>{
    let mean = x.mean();
    let variance = x.sub(mean).pow(2).div(x.shape[0]);
    return variance.sqrt();
  });
}

function variance(x){
  return tf.tidy(()=>{
    let mean = x.mean();
    let variance0 = x.sub(mean).pow(2).div(x.shape[0]);
    return variance0;
  });
}

function angular_resolution_loss(x, neighbors){
  return tf.tidy(()=>{
    var nV = x.shape[0];
    let losses = [];
    for(var i = 0; i<nV; i++){

      //stochastic
      // let dropoutRate = 0.1;
      // if(Math.random() < dropoutRate){
      //   continue;
      // }

      let n = neighbors[i].length;
      if(n <= 1){
        continue;
      }
      let v = x.gather(neighbors[i]).sub(x.gather(i));
      let xcoord = v.slice([0,0], [v.shape[0],1]);
      let ycoord = v.slice([0,1], [v.shape[0],1]);
      let norm = v.norm(2, 1, true);   

      // robjohn's answer at:
      // https://math.stackexchange.com/questions/1327253/how-do-we-find-out-angle-from-x-y-coordinates
      // TODO make it more stable (e.g. on path-10)
      let angle = tf.atan(ycoord.div(xcoord.add(norm))).mul(2).add(Math.PI).as1D();

      let angle_sorted = topk(angle, n);
      let angle_diff = angle_sorted
        .slice([0,], n-1)
        .sub(angle_sorted.slice([1,], n-1));
      let last = tf.scalar(Math.PI*2).add(angle_sorted.gather(n-1).sub(angle_sorted.gather(0)));
      angle_diff = tf.concat([angle_diff, last.reshape([1])]);

      let sensitivity = 1.0;
      let energy = tf.exp(angle_diff.mul(-1).mul(sensitivity)).sum();
      losses.push(energy);
      // 
      // let cross_entropy = angle_diff.div(Math.PI*2).log().sum().div(n).mul(-1/n); //cross entropy loss against uniform distribution={0.5, 0.5}
      // losses.push(cross_entropy);
    }
    if(losses.length > 0){
      let loss = tf.stack(losses).sum().mul(2);
      return loss;
    }else{
      return tf.scalar(0.0);
    }

  });
}


function angular_resolution_loss1(x, adj){
  return tf.tidy(()=>{
    
    let n = x.shape[0];
    let targetNodes = x.broadcastTo([x.shape[0], ...x.shape]);
    let sourceNodes = x.reshape([x.shape[0], 1, x.shape[1]])
    let v = targetNodes.sub(sourceNodes);//[n,n,2]

    let xcoord = v.slice([0,0,0], [v.shape[0],v.shape[1],1]);
    let ycoord = v.slice([0,0,1], [v.shape[0],v.shape[1],1]);
    let norm = v.norm('euclidean', 2, true);

    let angle = tf.atan(ycoord.div(xcoord.add(norm))).mul(2).add(Math.PI);
    angle = angle.reshape([angle.shape[0],angle.shape[1]]);
    angle = angle.mul(adj);
    angle = sort(angle);
    // console.log('angle');
    // let angle_diff = angle
    //     .slice([0], [angle.shape[0]-1])
    //     .sub(angle.slice([1], [angle.shape[0]-1]));
    // angle_diff.print();
    // let energy = tf.exp(angle_diff.abs().mul(-1)).mean().mul(0);
    // energy.print();
    // return tf.scalar(0.0);
    return angle.mean();
  });
}


function sign(p, line){
  let s = (p.x-line[0].x)*(line[1].y-line[0].y) + (p.y-line[0].y)*(line[0].x-line[1].x);
  if(s>0){
    return 1;
  }else if(s==0){
    return 0;
  }else{
    return -1;
  }
}


function isSameSide(p1, p2, line){
  let s1 = sign(p1, line);
  let s2 = sign(p2, line);
  return s1 * s2;
}


function hasCrossing(nodes, includeEndpoints=false){
  let [n1,n2,n3,n4] = nodes;
  let ss1 = isSameSide(n1, n2, [n3, n4]);
  let ss2 = isSameSide(n3, n4, [n1, n2]);
  if (includeEndpoints){
    return ss1<=0 && ss2 <= 0;
  }else{
    return ss1<0 && ss2 < 0;
  }
}


function dot(x, y){
  let dim = 1;
  let keepDims=true;
  return tf.tidy(()=>{
    return x.mul(y).sum(dim, keepDims);
  });
}


function cosSimilarity(x, y){
  let dim = 1;
  let keepDims = true;
  return tf.tidy(()=>{
    let xNorm = x.norm('euclidean', dim, keepDims);
    let yNorm = y.norm('euclidean', dim, keepDims);
    let cos = dot(x,y).div(xNorm.mul(yNorm));
    return cos
  });
}


function graph2crossings(graph){
  let crossings = [];
  for (let i=0; i<graph.edges.length; i++){
    let e1 = graph.edges[i];
    for (let j=i+1; j<graph.edges.length; j++){
      let e2 = graph.edges[j];
      let a = e1.source.index;
      let b = e1.target.index;
      let c = e2.source.index;
      let d = e2.target.index;
      let nodes = [graph.nodes[a], 
      graph.nodes[b], 
      graph.nodes[c], graph.nodes[d]];
      let crossed = hasCrossing(nodes);
      if (crossed){
        crossings.push([[a,b],[c,d]]);
      }
    }
  }
  return crossings;
}

function crossing_angle_metric(x, graph){
  // return 0.0; //disabled for faster training

  let crossings = graph2crossings(graph);
  if(crossings.length>0){
    //metric
    let x_data = x.arraySync();
    let p1 = crossings.map(d=>x_data[d[0][0]]);
    let p2 = crossings.map(d=>x_data[d[0][1]]);
    let p3 = crossings.map(d=>x_data[d[1][0]]);
    let p4 = crossings.map(d=>x_data[d[1][1]]);
    let e1 = numeric.sub(p1, p2);
    let e2 = numeric.sub(p3, p4);
    let cos = zip(e1, e2).map((d)=>{
      let e1 = d[0];
      let e2 = d[1];
      let cos = numeric.dot(e1, e2) / (numeric.norm2(e1) * numeric.norm2(e2));
      return cos;
    })
    let angle_normed = cos.map(c=>{
      return Math.abs(Math.acos(c) - Math.PI/2) / Math.PI/2;
    });
    let metric = math.max(angle_normed);
    return metric;
  }else{
    return 0.0;
  }
}


function crossing_angle_loss(x, graph, sampleSize=1){
  let crossings = graph2crossings(graph);

  sampleSize = Math.min(5, Math.ceil(crossings.length)/2);
  let sampledCrossings = _.sample(crossings, sampleSize);
  // sampledCrossings = crossings;

  if(sampledCrossings.length > 0){

    let loss = tf.tidy(()=>{
      let p1 = x.gather( sampledCrossings.map(d=>d[0][0]) );
      let p2 = x.gather( sampledCrossings.map(d=>d[0][1]) );
      let p3 = x.gather( sampledCrossings.map(d=>d[1][0]) );
      let p4 = x.gather( sampledCrossings.map(d=>d[1][1]) );
      let e1 = p2.sub(p1);
      let e2 = p4.sub(p3);
      let cos = cosSimilarity(e1, e2);
      let loss = cos.square().sum().div(10);
      return loss;
    });

    //metric
    let x_data = x.arraySync();
    let p1 = crossings.map(d=>x_data[d[0][0]]);
    let p2 = crossings.map(d=>x_data[d[0][1]]);
    let p3 = crossings.map(d=>x_data[d[1][0]]);
    let p4 = crossings.map(d=>x_data[d[1][1]]);
    let e1 = numeric.sub(p1, p2);
    let e2 = numeric.sub(p3, p4);
    let cos = zip(e1, e2).map((d)=>{
      let e1 = d[0];
      let e2 = d[1];
      let cos = numeric.dot(e1, e2) / (numeric.norm2(e1) * numeric.norm2(e2));
      return cos;
    })
    let angle_normed = cos.map(c=>{
      return Math.abs(Math.acos(c) - Math.PI/2) / Math.PI/2;
    });
    let metric = math.max(angle_normed);
    return [loss, metric];
  }else{
    return [tf.scalar(0.0), 0.0];
  }
}


// https://github.com/tensorflow/tfjs/issues/1601
function topk(x, k){
  const shape = x.shape.slice(0,-1);
  const [n] = x.shape.slice(  -1),
         m  =   shape.reduce( (k,l) => k*l, 1 );
  const data = x.dataSync(),
      sorted = new Int32Array(n),
     indices = new Int32Array(m*k);
  for( let i=0; i < m; ++i ){
    for( let j=0; j < n; j++ )
      sorted[j] = j + i*n;
    sorted.sort( (i,j) => data[j] - data[i] );
    for( let j=0; j < k; j++ )
      indices[j + i*k] = sorted[j];
  }
  shape.push(k);
  return x.flatten()
          .gather(indices)
          .reshape(shape);
}

function sort(x){
  //sort each row of a matrix x independently, remove non-positives
  const nrows = x.shape[0];
  const nentries = x.shape[1];
  const data = x.arraySync();
  const indices = d3.range(nentries);

  let final_indices = [];
  for(let i=0; i<nrows; i++){
    let row = data[i];
    let pairs = zip(indices, row);
    let pairs_sorted = pairs.sort((a,b)=>b[1]-a[1]);
    let indices_sorted = pairs_sorted.filter(d=>d[1]>0).map(d=>i*nrows + d[0]);
    final_indices = final_indices.concat(indices_sorted);
  }
  return x.flatten()
          .gather(final_indices);
}


// const customSqrt = tf.customGrad((x, save) => {
//   save([x]);
//   // Override gradient of our custom x ^ 2 op to be dy * abs(x);
//   return {
//     value: x.square(), 
//     // gradFunc: dy => {[dy.mul(2).mul(save[0].notEqual(0.0))]},
//     gradFunc: (dy, saved) => {
//       let x = saved[0];
//       return [
//         x
//         .mul(x.greater(0.0).toFloat())
//         .sqrt()
//         .reciprocal()
//         .mul(0.5)
//         .clipByValue(0,1000.0),
//       ];
//     },
//   };
// });
// let x = tf.tensor1d([-1, -2, 0, 1, 2]);
// let dx = tf.grad(x => customSqrt(x));


function computeThreshScaleMargin(pdist, n_neighbors){
  let n_nodes = pdist.length;
  let thresh = [];
  let scale = [];
  let margin = [];

  for(let i=0; i<n_nodes; i++){
    let dist = pdist[i];
    let k = n_neighbors[i];
    let sorted = dist.sort((a,b)=>a-b);
    let th = undefined;
    if(k == n_nodes-1){
      th = sorted[n_nodes-1] + 1;
    }else{
      th = (sorted[k] + sorted[Math.min(k+1, n_nodes-1)])/2;
    }
    let ma = sorted[Math.min(k+1, n_nodes-1)] - sorted[k];
    thresh.push(th);
    scale.push(sorted[1]);
    margin.push(ma);
  }
  
  return [
    tf.tensor(thresh, [n_nodes, 1]), 
    tf.tensor(scale, [n_nodes, 1]),
    tf.tensor(margin, [n_nodes, 1]),
  ];
}




function cumulativeJaccardLoss(pred, truth){
  // pred, truth: 1D/2D js {0,1}-arrays
  // return: an array of intermediate values of the cumulative jaccard loss , where jaccard_loss := 1-jaccard_index;
  pred = pred.flat();
  truth = truth.flat();
  let positives = math.sum(truth);
  let res = [];
  for(let i=0; i<pred.length; i++){
    let misclassified = i;
    let loss = misclassified/(misclassified+positives);
    res.push(loss);
  }
  return res;
}


function zip(){
  let a = arguments[0];
  let res = [];
  for(let i=0; i<a.length; i++){
    res.push(
      d3.range(arguments.length).map((j)=>arguments[j][i])
    );
  }
  return res;
}


function jaccardIndex(pred, truth, thresh=0.0){
  pred = pred.flat();
  truth = truth.flat();
  let cum_intersect = 0;
  let cum_union = 0;
  for(let i=0; i<pred.length; i++){
    let interect = (pred[i]>thresh && truth[i]==1) ? 1:0;
    let union = (pred[i]>thresh || truth[i]==1) ? 1:0;
    cum_intersect += interect;
    cum_union += union;
  }
  let jaccard = 0;
  if(cum_union > 0){
    jaccard = cum_intersect/cum_union;
  }
  return jaccard;
}


function jaccardMargins(error, predicted_labels, truth){
  let error_index_pairs = zip(error, d3.range(error.length));
  let error_index_pairs_sorted = error_index_pairs.sort((a,b)=>b[0]-a[0]);
  let error_sorted = error_index_pairs_sorted.map(d=>d[0]);
  let index_sorted = error_index_pairs_sorted.map(d=>d[1]);
  let predicted_labels_sorted = index_sorted.map((i)=>predicted_labels[i]);
  let truth_sorted = index_sorted.map((i)=>truth[i]);

  let cum_jaccard_loss_sorted = cumulativeJaccardLoss(predicted_labels_sorted, truth_sorted);
  let jaccard_margins_sorted = cum_jaccard_loss_sorted.map((d,i)=>{
    if(i==0){
      return d;
    }else{
      return (d - cum_jaccard_loss_sorted[i-1]);
    }
  });
  let margins = zip(index_sorted, jaccard_margins_sorted).sort((a,b)=>a[0]-b[0]).map(d=>d[1]);
  return margins;
}


function lovaszHingeLoss(pred, truth){
  //pred: 1D/2D tf vector
  //truth: 1D/2D js array
  //return: a loss (with type=tf.scalar) that is differentiable wrt pred (and thus x)
  //
  return tf.tidy(()=>{
    let n = truth.length;
    let eye = tf.eye(pred.shape[0]);
    pred = pred.mul(tf.scalar(1.0).sub(eye)).sub(eye); //never consider the diagonal entries
    // pred = pred.mul(tf.scalar(1.0).sub(eye)); //never consider the diagonal entries
    
    pred = pred.reshape([-1]);
    truth = truth.flat();

    //hinge
    let error = tf.scalar(1.0).sub(
      tf.mul( pred, tf.tensor(truth).mul(2).sub(1) )
    ).relu();
    let predicted_labels = pred.arraySync().map(d=>d>0.0 ? 1.0 : 0.0);

    //softmax
    // pred = pred.sigmoid();
    // let error = pred.sub(truth).abs();
    // let predicted_labels = pred.arraySync().map(d=>d>0.5 ? 1.0 : 0.0);

    let coef = jaccardMargins(error.arraySync(), predicted_labels, truth);
    return error.dot(coef);
  });
  
  
}


function neighbor_loss(pdist, adj, thresh, scale, margin){
  return tf.tidy(()=>{
    let truth = adj;
    let mask = tf.scalar(1.0).sub(tf.eye(pdist.shape[0]));
    // let pred = pdist.sub( thresh ).mul(-1);
    
    // let pred = pdist.sub( thresh.clipByValue(0.2, 1.0) ).mul(-1);
    let pred = pdist
    .sub( thresh.clipByValue(0.01, 50.0) )
    .div( thresh.clipByValue(0.01, 50.0) )
    .mul(-1);
    let loss = lovaszHingeLoss(pred, truth.arraySync());


    let pred2 = pdist.sub(thresh).mul(-1).mul(mask);
    let metric = jaccardIndex(pred2.arraySync(), truth.arraySync());
    return [loss, metric];
  });
}


function center_loss(x, center){
  return tf.tidy(()=>{
    center = tf.tensor(center);
    return x.mean(0).sub(center).div(center).pow(2).sum();
  });
}


function boundary_loss(xy, xyMin, xyMax){
  return tf.tidy(()=>{
    let lossLeft = xy.sub(tf.tensor(xyMin)).mul(-1).relu().pow(2).sum();
    let lossRight = xy.sub(tf.tensor(xyMax)).relu().pow(2).sum();
    return lossLeft.add(lossRight);
  });
}

let boundaries = undefined;
let boundary_lr = 0.15;
let boundary_optim = undefined;

function crossing_number_loss(x, edgePairs){
  let [edges1, edges2] = edgePairs;

  //stochastic
  // let samples = new Set(_.sample(d3.range(edges1.length), 10));
  // edges1 = edges1.filter((d,i)=>samples.has(i));
  // edges2 = edges2.filter((d,i)=>samples.has(i));
  // boundaries = tf.variable(
  //   tf.randomUniform([edges1.length, 3, 1], -0.1, 0.1));
  // boundary_optim = tf.train.adam(boundary_lr);


  if(boundaries === undefined){
    boundaries = tf.variable(
      tf.randomUniform([edges1.length, 3, 1], -0.1, 0.1));
    boundary_optim = tf.train.adam(boundary_lr);
  }

  // inner training loop: (E step in EM)
  // find optimal decision boundaries given current x
  for(let b_iter = 0; b_iter<1; b_iter++){
    let l = boundary_optim.minimize(()=>{
      let ones = tf.ones([x.shape[0], 1]);
      let x1 = tf.concat([x, ones], 1);
      e1 = x1.gather(edges1);
      e2 = x1.gather(edges2);

      let pred1 = e1.matMul(boundaries);
      let pred2 = e2.matMul(boundaries);

      //svm loss = mean(relu(1 - pred*target)) + lambda * ||w||^2; we pick lambda = 0 (hard boundary)
      let loss1 = tf.relu(tf.scalar(1).sub(pred1.mul(1))).sum(); 
      let loss2 = tf.relu(tf.scalar(1).sub(pred2.mul(-1))).sum();
      let margin = boundaries.slice([0,0,0], [boundaries.shape[0], 2, 1]).norm(2,0).sum();
      let loss = loss1.add(loss2).add(margin.mul(0.05));

      return loss;
    }, true, [boundaries,]);
  }

  // outer loss (loss for M step)
  return tf.tidy(()=>{

    let ones = tf.ones([x.shape[0], 1]);
    let x1 = tf.concat([x, ones], 1);


    e1 = x1.gather(edges1);
    e2 = x1.gather(edges2);
  
    let pred1 = e1.matMul(boundaries);
    let pred2 = e2.matMul(boundaries);

    //svm loss = mean(relu(1 - pred*target)) + lambda * ||w||^2; we pick lambda = 0 (hard boundary)
    let loss1 = tf.relu(tf.scalar(1).sub(pred1.mul(1))).sum(); 
    let loss2 = tf.relu(tf.scalar(1).sub(pred2.mul(-1))).sum();
    let loss = loss1.add(loss2);

    return loss.mul(0.01);
  });
}


function gabriel_loss(x, adj, pdist){
  return tf.tidy(()=>{

    let x1 = x.broadcastTo([x.shape[0], ...x.shape]);
    let centers = x1.add(x1.transpose([1,0,2])).div(2.0);
    centers = centers.broadcastTo([x.shape[0], ...centers.shape]);
    let dist = centers.sub(x.reshape([x.shape[0], 1, 1, x.shape[1]])).norm('euclidean', 3);//[x, e1, e2]

    let radii = pdist.div(2.0);//[e1, e2]

    let loss = dist.sub(radii).mul(-1).relu();
    let mask = adj;
    loss = loss.mul(mask);
    loss = loss.sum().div(10);

    let metric = tf.scalar(1.0).sub( dist.sub(radii).mul(-1).relu().div(radii.add(tf.eye(radii.shape[0]))) );
    metric = metric.mul(mask).add(tf.scalar(1.0).sub(mask).mul(1e6)).min();

    return [loss, metric.dataSync()[0]];
  });
}


function upwardness_loss(x, graph){
  let sourceIndices, targetIndices;
  if('sourceIndices' in graph){
    sourceIndices = graph.sourceIndices;
    targetIndices = graph.targetIndices;
  }else{
    sourceIndices = graph.edges.map(e=>e.source.index);
    targetIndices = graph.edges.map(e=>e.target.index);
    graph.sourceIndices = sourceIndices;
    graph.targetIndices = targetIndices;
  }
  let edgeCount = graph.edges.length;

  return tf.tidy(()=>{
    let source = x.gather(sourceIndices);
    let target = x.gather(targetIndices);
    let dir = target.sub(source);
    let y = dir.slice([0,1], [edgeCount, 1]);
    let loss = y.sub(1).mul(-1).relu();
    loss = tf.add(loss.pow(2).sum().mul(0.7), loss.sum().mul(0.3));//elastic-net loss
    return [loss, 0.0];
  });
}



// function gabriel_loss(x, adj, pdist){
//   return tf.tidy(()=>{

//     let x1 = x.broadcastTo([x.shape[0], ...x.shape]);
//     let centers = x1.add(x1.transpose([1,0,2])).div(2.0);
//     centers = centers.broadcastTo([x.shape[0], ...centers.shape]);
//     let dist = centers.sub(x.reshape([x.shape[0], 1, 1, x.shape[1]])).norm('euclidean', 3);//[x, e1, e2]
//     let radii = pdist.div(2.0);//[e1, e2]

//     let mask = adj;
//     let mask2 = tf.scalar(1.0).sub(tf.eye(adj.shape[0]));
//     let pred = dist.div( radii.add(tf.eye(radii.shape[0])) ).sub(1.0);

//     //softmin
//     // pred = pred
//     // .mul(mask)
//     // .mul(mask2.reshape([mask2.shape[0], 1, mask2.shape[1]]))
//     // .mul(mask2.reshape([mask2.shape[0], mask2.shape[1], 1]));
//     // let weights = softmin(pred, [0], 1);
//     // let non_zero = tf.tensor(pred.greater(0).arraySync());
//     // weights = weights.div(weights.mul(non_zero).sum(0).add(0.0001));
//     // pred = pred.mul(weights);
//     // pred = pred.sum(0);
//     // 
    
//     pred = pred
//     .mul(mask)
//     .min(0);


//     let loss = lovaszHingeLoss(pred, adj.arraySync());
//     loss = loss.div(5);

//     let metric = tf.scalar(1.0).sub(  dist.sub(radii).mul(-1).relu().div(radii.add(0.001)) );
//     metric = metric.mul(mask).add(tf.scalar(1.0).sub(mask).mul(1e2)).min();
//     return [loss, metric.dataSync()[0]];
//   });
// }


function rotations(n=10){
  return tf.tidy(()=>{
    let theta = tf.linspace(0, Math.PI*2, n+1).slice([0], [n]);
    let cos = tf.cos(theta);
    let sin = tf.sin(theta);
    return tf.stack([cos, sin, sin.mul(-1), cos], -1).reshape([n,2,2]);
  });
}


function softmax(x, dims=[0], sensitivity=1.0){
  return tf.tidy(()=>{
    let y = x.sub(x.max());
    let exp = tf.exp(y.mul(sensitivity));

    let sum = exp;
    for(let d of dims){
      sum = sum.sum(d,true);
    }
    let res = exp.div(sum);
    return res;
  });
}

function softmin(x, dims=[0], sensitivity=1.0){
  return softmax(x.mul(-1));
}



function aspect_ratio_loss(x){
  return tf.tidy(()=>{
    let n = 7;
    let rot = rotations(n);
    x = x.broadcastTo([n,...x.shape]).matMul(rot);
    let maxWeight = softmax(x, [0,2], 0.1);
    let minWeight = softmax(x.mul(-1), [0,2], 0.1);
    let max = x.mul(maxWeight).sum(1);
    let min = x.mul(minWeight).sum(1);
    let size = max.sub(min).add(0.3); // add a small constant for stability on cube & path-10;
    let prob = size.div(size.sum(1, true)); //n two-category probabilities;
    let cross_entropy = prob.log().sum().div(n).mul(-0.5); //cross entropy loss against uniform distribution={0.5, 0.5}
    let loss = cross_entropy.mul(500.0);

    let max_for_metric = x.max(1);//shape=[7,2]
    let min_for_metric = x.min(1);
    let size_for_metric = max_for_metric.sub(min_for_metric);
    let width_for_metric = size_for_metric.slice([0,0],[n,1]);
    let height_for_metric = size_for_metric.slice([0,1],[n,1]);
    let ratio_for_metric = tf.concat([
      width_for_metric.div(height_for_metric),
      height_for_metric.div(width_for_metric),
    ]);
    let metric = ratio_for_metric.min().dataSync()[0];

    return [loss, metric];
  });
}

// function area_loss(x, pdist, edges, desiredArea){
//   if(desiredArea===undefined){
//     // desiredArea = Math.sqrt(edges.length) / 4;
//     desiredArea = 2;
//     // desiredArea = pdist.max();
//     console.log(desiredArea);
//   }
//   let minEdgeLength = math.min(pdist.arraySync().flat().filter(d=>d>0.001));
//   return tf.tidy(()=>{
//     let scale = Math.max(2.0, minEdgeLength);
//     let min = x.min(0);
//     let max = x.max(0);

//     // min = tf.scalar(1.0).div(x.sub(min).add(0.01)).mul(x).mean(0);
//     // max = tf.scalar(1.0).div(max.sub(x).add(0.01)).mul(x).mean(0);
    
//     let sideLengths = max.sub(min).div(scale);
//     let area = sideLengths.gather(0).mul(sideLengths.gather(1));
//     let loss = area.sub(desiredArea).pow(2);
//     return loss;
//   });
// }
// 

function graph2neighbors(graph){
  let res = {};
  for (let e of graph.edges){
    if(res[e.source.index] === undefined){
      res[e.source.index] = [e.target.index,];
    }else{
      res[e.source.index].push(e.target.index);
    }

    if(res[e.target.index] === undefined){
      res[e.target.index] = [e.source.index,];
    }else{
      res[e.target.index].push(e.source.index);
    }
  }
  return res;
}


function crossing_number_metric(x, graph){
  let n = x.shape[0];
  let crossings = graph2crossings(graph);
  let metric = crossings.length;
  return metric;
}

function trainOneIter(dataObj, optimizer, computeMetric=true){
  let x = dataObj.x;
  // let x_array = x.arraySync();
  let graphDistance = dataObj.graphDistance;
  let stressWeight = dataObj.stressWeight
  let graph = dataObj.graph;
  let n_neighbors = dataObj.n_neighbors;
  let adj = dataObj.adj;
  let coef = dataObj.coef;
  let metrics = {};
  let loss = optimizer.minimize(()=>{
    let pdist = pairwise_distance(x);
    let loss = tf.tidy(()=>{

      let vmin = [0, 0];
      let vmax = [dataObj.graph.width, dataObj.graph.height];
      let l = center_loss(x, graph.center);
      // .add(boundary_loss(x, vmin, vmax));
      // let l = boundary_loss(x, vmin, vmax);
      // let l = tf.scalar(0);

      if(coef.stress > 0){
        let [st, m_st, pdist_normalized] = stress_loss(pdist, graphDistance, stressWeight);
        metrics.stress = m_st;
        metrics.pdist = pdist_normalized;
        l = l.add(st.mul(coef.stress));
      }else if(computeMetric){
        let [st, m_st, pdist_normalized] = stress_loss(pdist, graphDistance, stressWeight);
        metrics.stress = m_st;
        metrics.pdist = pdist_normalized;
      }

      if(coef.crossing_angle > 0){
        let [an, m_an] = crossing_angle_loss(x, graph, dataObj.sampleSize || 1);
        metrics.crossing_angle = m_an;
        l = l.add(an.mul(coef.crossing_angle));
      }else{
        let m_an = crossing_angle_metric(x, graph);
        metrics.crossing_angle = m_an;
      }
      
      if (coef.neighbor > 0){
        let [thresh, scale, margin] = computeThreshScaleMargin(pdist.arraySync(), n_neighbors);
        let [nb, m_nb] = neighbor_loss(pdist, adj, thresh, scale, margin);
        metrics.neighbor = m_nb;
        if (coef.neighbor > 0){
          l = l.add(nb.mul(coef.neighbor));
        }
      }else if (computeMetric){
        let [thresh, scale, margin] = computeThreshScaleMargin(pdist.arraySync(), n_neighbors);
        let truth = adj;
        let mask = tf.scalar(1.0).sub(tf.eye(pdist.shape[0]));
        let pred = pdist.sub(thresh).mul(-1).mul(mask);
        let m_nb = jaccardIndex(pred.arraySync(), truth.arraySync());
        metrics.neighbor = m_nb;
      }

      
      if(coef.edge_uniformity > 0){
        let [eu, m_eu] = edge_uniformity_loss(pdist, adj);
        metrics.edge_uniformity = m_eu;
        l = l.add(eu.mul(coef.edge_uniformity));
      }

      
      if (coef.crossing_number > 0){
        let m_cs = crossing_number_metric(x, graph);
        metrics.crossing_number = m_cs;
        let cs = crossing_number_loss(x, dataObj.edgePairs);
        l = l.add(cs.mul(coef.crossing_number));
      }


      if(coef.angular_resolution > 0){
        let m_ar = angular_resolution_metric(x, dataObj.neighbors);
        metrics.angular_resolution = m_ar;

        if(coef.angular_resolution > 0){
          let ar = angular_resolution_loss(x, dataObj.neighbors);
          l = l.add(ar.mul(coef.angular_resolution));
        }
      }

      
      if(coef.vertex_resolution > 0){
        let [vr, m_vr] = vertex_resolution_loss(pdist);
        metrics.vertex_resolution = m_vr;
        if(coef.vertex_resolution > 0){
          l = l.add(vr.mul(coef.vertex_resolution));
        }
      }


      if(coef.aspect_ratio > 0){
        let [as, m_as] = aspect_ratio_loss(x);
        metrics.aspect_ratio = m_as;
        l = l.add(as.mul(coef.aspect_ratio));
      }


      if(coef.gabriel > 0){
        let [gb, m_gb] = gabriel_loss(x, adj, pdist);
        metrics.gabriel = m_gb;
        l = l.add(gb.mul(coef.gabriel));
      }

      if(coef.upwardness > 0){
        let [up, m_up] = upwardness_loss(x, graph);
        metrics.upwardness = m_up;
        l = l.add(up.mul(coef.upwardness));
      }

      // let upperBound = 1e4;
      // return l.div(upperBound).tanh().mul(upperBound);
      return l;
    });
    return loss;
  }, true, [x]);
  return {loss, metrics};

}


function train(dataObj, remainingIter, optimizers, callback){
  if (remainingIter <= 0 || !isPlaying){
    cancelAnimationFrame(dataObj.animId);
    console.log('Max iteration reached, please double click the play button to restart');
    window.playButton.on('click')(false);
  }else{
    let computeMetric = true;//remainingIter % 50 == 0;
    let {loss, metrics} = trainOneIter(dataObj, optimizers[0], computeMetric);
    if (callback){
      callback({
        remainingIter,
        loss: loss.dataSync()[0],
        metrics
      });
    }
    dataObj.animId = requestAnimationFrame((t)=>{
      if(this.t !== undefined){
        let dt = t - this.t;
        let fps = 1000 / dt;
        // console.log(fps.toFixed(2));
      }
      train(dataObj, remainingIter-1, optimizers, callback);
      this.t = t;
    });
  }
}



function getEdgePairs(graph){
  let edges1 = [];
  let edges2 = [];
  for(let edge1 of graph.edges){
    for(let edge2 of graph.edges){
      let e1 = [edge1.source.index, edge1.target.index];
      let e2 = [edge2.source.index, edge2.target.index];
      let not_incident = (new Set(e1.concat(e2))).size == 4;
      if(not_incident && e1[0] < e2[0]){
        edges1.push(e1);
        edges2.push(e2);

      }
    }
  }
  return [edges1, edges2];
}
