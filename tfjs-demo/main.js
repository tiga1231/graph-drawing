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


function stress_loss(pred, target, weight){
  return tf.tidy(()=>{
    let stress = pred.sub(target).square().mul(weight).sum().div(4);
    return [stress, stress.mul(-1).add(1.0)];
  });
}


function edge_uniformity_loss(pdist, adj){
  let n_nodes = pdist.shape[0];
  let N = adj.sum(); // number of edges (*2)
  return tf.tidy(()=>{
    let mean = pdist.mul(adj).sum().div(N);

    let loss = pdist
    .sub(mean.add(0.1)) //for stability
    .mul(adj)
    .square()
    .sum()
    // .div(N)
    .div(10)
    .sqrt();

    let metric = pdist
    .sub(mean)
    .mul(adj)
    .square()
    .sum()
    .div(N)
    .sqrt()
    .mul(-1)
    .add(1.0);

    return [loss, metric];
  });
}


function vertex_resolution_loss(pdist, resolution=0.1){
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
    return loss;
  });
}

function angular_resolution_loss0(x, neighbors){
  return tf.tidy(()=>{
    var nV = x.shape[0];
    let losses = [];
    for(var i = 0; i<nV; i++){
      let n = neighbors[i].length;
      let v = x.gather(neighbors[i]).sub(x.gather(i));
      let xcoord = v.slice([0,0], [v.shape[0],1]);
      let ycoord = v.slice([0,1], [v.shape[0],1]);
      let norm = v.norm(2, 1, true);   

      // robjohn's answer at:
      // https://math.stackexchange.com/questions/1327253/how-do-we-find-out-angle-from-x-y-coordinates
      let angle = tf.atan(ycoord.div(xcoord.add(norm))).mul(2).add(Math.PI).as1D();

      let angle_sorted = topk(angle, n);
      let angle_diff = angle_sorted
        .slice([0,], n-1)
        .sub(angle_sorted.slice([1,], n-1));
      let last = tf.scalar(3.1415926*2).add(angle_sorted.gather(n-1).sub(angle_sorted.gather(0)));
      angle_diff = tf.concat([angle_diff, last.reshape([1])]);
      let energy = tf.exp(angle_diff.mul(-1)).sum();
      losses.push(energy);
    }
    let loss = tf.stack(losses).sum();
    return loss;
  });
}


function angular_resolution_loss(x, adj){
  return tf.tidy(()=>{
    x.print();
    let n = x.shape[0];
    let targetNodes = x.broadcastTo([x.shape[0], ...x.shape]);
    let sourceNodes = x.reshape([x.shape[0], 1, x.shape[1]])
    let v = targetNodes.sub(sourceNodes);//[n,n,2]

    let xcoord = v.slice([0,0,0], [v.shape[0],v.shape[1],1]);
    let ycoord = v.slice([0,0,1], [v.shape[0],v.shape[1],1]);
    let norm = v.norm('euclidean', 2, true);
    norm.print();
    let angle = tf.atan(ycoord.div(xcoord.add(norm))).mul(2).add(Math.PI);
    angle = angle.reshape([angle.shape[0],angle.shape[1]]);
    angle = angle.mul(adj);
    angle = sort(angle);
    console.log(angle.shape);
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


function crossing_angle_loss(x, graph, sampleSize=1){
  let crossings = [];

  for (let e1 of graph.edges){
    for (let e2 of graph.edges){
      let i = e1.source.index;
      let j = e1.target.index;
      let k = e2.source.index;
      let l = e2.target.index;
      let nodes = [graph.nodes[i], graph.nodes[j], graph.nodes[k], graph.nodes[l]];
      let crossed = hasCrossing(nodes);
      if (crossed){
        crossings.push([[i,j],[k,l]]);
      }
    }
  }

  sampleSize = crossings.length;
  // // let sampleSize = Math.min(5, Math.ceil(crossings.length)/2);
  let sampledCrossings = _.sample(crossings, sampleSize);
  if(sampledCrossings.length > 0){
    return tf.tidy(()=>{
      let p1 = x.gather( sampledCrossings.map(d=>d[0][0]) );
      let p2 = x.gather( sampledCrossings.map(d=>d[0][1]) );
      let p3 = x.gather( sampledCrossings.map(d=>d[1][0]) );
      let p4 = x.gather( sampledCrossings.map(d=>d[1][1]) );
      let e1 = p2.sub(p1);
      let e2 = p4.sub(p3);
      let cos = cosSimilarity(e1, e2);
      return [cos.square().mean(), tf.scalar(1.0).sub(cos.acos().min().sub(Math.PI/2).abs().div(Math.PI/2))];
    });
  }else{
    return [tf.scalar(0.0), tf.scalar(0.0)];
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
    let sorted = dist.sort();
    let th = undefined;
    if(k == n_nodes){
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
    tf.tensor(thresh, [n_nodes,1]), 
    tf.tensor(scale, [n_nodes,1]),
    tf.tensor(margin, [n_nodes,1]),
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
    let pred = pdist.sub( thresh.clipByValue(0.2, 1.0) ).mul(-1);
    let loss = lovaszHingeLoss(pred, truth.arraySync());

    let pred2 = pdist.sub(thresh).mul(-1).mul(mask);
    let metric = jaccardIndex(pred2.arraySync(), truth.arraySync());
    return [loss, metric];
  });
}


function center_loss(x){
  return tf.tidy(()=>{
    return x.mean(0).pow(2).sum();
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


function gabriel_loss(x,adj, pdist){
  return tf.tidy(()=>{

    let x1 = x.broadcastTo([x.shape[0], ...x.shape]);
    let centers = x1.add(x1.transpose([1,0,2])).div(2.0);
    centers = centers.broadcastTo([x.shape[0], ...centers.shape]);
    let dist = centers.sub(x.reshape([x.shape[0], 1, 1, x.shape[1]])).norm('euclidean', 3);
    let radii = pdist.div(2.0);
    let loss = dist.sub(radii).mul(-1).relu().pow(2);
    let mask = adj;
    loss = loss.mul(mask);
    loss = loss.sum().div(2);
    return loss;
  });
  
}

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


function aspect_ratio_loss(x){
  return tf.tidy(()=>{
    let n = 7;
    let rot = rotations(n);
    x = x.broadcastTo([n,...x.shape]).matMul(rot);
    let maxWeight = softmax(x, [0,2], 0.1);
    let minWeight = softmax(x.mul(-1), [0,2], 0.1);
    let max = x.mul(maxWeight).sum(1);
    let min = x.mul(minWeight).sum(1);
    let size = max.sub(min).add(1); // add 1 for stability on cube & path-10;
    let prob = size.div(size.sum(1, true)); //n two-category probabilities;
    let cross_entropy = prob.log().sum().div(n).mul(-0.5); //cross entropy loss against uniform distribution={0.5, 0.5}
    return cross_entropy.mul(500.0);
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


function trainOneIter(dataObj, optimizer){
  let x = dataObj.x;
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
      let l = center_loss(x);
      if(coef.stress > 0){
        let [st, metric] = stress_loss(pdist, graphDistance, stressWeight);
        l = l.add(st.mul(coef.stress));
        metrics.stress = metric.dataSync()[0];
      }else{
        // let st = stress_loss(pdist, graphDistance, stressWeight);
        // metrics.stress = -st.dataSync()[0];
      }

      if(coef.crossing_angle > 0){
        let [an, metric] = crossing_angle_loss(x, graph, dataObj.sampleSize || 1);
        l = l.add(an.mul(coef.crossing_angle));
        metrics.angle = metric.dataSync()[0];
      }

      if (coef.neighbor > 0){
        // let nb = neighbor_loss(pdist, adj, n_neighbors, 1000);
        let [thresh, scale, margin] = computeThreshScaleMargin(pdist.arraySync(), n_neighbors);
        let [nb, metric] = neighbor_loss(pdist, adj, thresh, scale, margin);
        l = l.add(nb.mul(coef.neighbor));
        metrics.neighbor = metric;
      }

      if(coef.edge_uniformity > 0){
        let [eu, metric] = edge_uniformity_loss(pdist, adj);
        l = l.add(eu.mul(coef.edge_uniformity));
        metrics.edge_uniformity = metric.dataSync()[0];
      }

      if (coef.crossing_number > 0){
        let cs = crossing_number_loss(x, dataObj.edgePairs);
        l = l.add(cs.mul(coef.crossing_number));
      }
      if(coef.angular_resolution > 0){
        let ar = angular_resolution_loss0(x, dataObj.neighbors);
        // let ar = angular_resolution_loss(x, adj);
        l = l.add(ar.mul(coef.angular_resolution));
      }
      if(coef.area > 0){
        let al = area_loss(x, pdist, dataObj.edges, dataObj.desiredArea);
        l = l.add(al.mul(coef.area));
      }
      if(coef.vertex_resolution > 0){
        let vr = vertex_resolution_loss(pdist);
        l = l.add(vr.mul(coef.vertex_resolution));
      }
      if(coef.aspect_ratio > 0){
        let as = aspect_ratio_loss(x);
        l = l.add(as.mul(coef.aspect_ratio));
      }
      if(coef.gabriel > 0){
        let gb = gabriel_loss(x, adj, pdist);
        l = l.add(gb.mul(coef.gabriel));
      }

      let upperBound = 1e4;
      return l.div(upperBound).tanh().mul(upperBound);
    });
    return loss;
  }, true, [x]);
  console.log(metrics);
  return {loss,metrics};

}


function train(dataObj, remainingIter, optimizers, callback){
  if (remainingIter <= 0 || !isPlaying){
    cancelAnimationFrame(dataObj.animId);
    console.log('Max iteration reached, please double click the play button to restart');
    window.playButton.on('click')(false);
  }else{
    let {loss, metrics} = trainOneIter(dataObj, optimizers[0]);
    

    if (callback){
      callback({
        remainingIter,
        loss: loss.dataSync()[0],
      });
    }
    dataObj.animId = requestAnimationFrame(()=>{
      train(dataObj, remainingIter-1, optimizers, callback);
    });
  }
}


function updateAxes(svg, sx, sy){
  let ax = d3.axisBottom(sx)
  .tickSizeInner(-(sy.range()[0]- sy.range()[1]));
  let ay = d3.axisLeft(sy)
  .tickSizeInner(-(sx.range()[1]- sx.range()[0]));
  let gx = svg.selectAll('.xAxis')
  .data([0,])
  .enter()
  .append('g')
  .attr('class', 'xAxis');
  gx = svg.selectAll('.xAxis')
  .attr('transform', `translate(${0},${sy.range()[0]})`)
  .call(ax);
  let gy = svg.selectAll('.yAxis')
  .data([0,])
  .enter()
  .append('g')
  .attr('class', 'yAxis');
  gy = svg.selectAll('.yAxis')
  .attr('transform', `translate(${sx.range()[0]},${0})`)
  .call(ay);
}

function traceLoss(svg, losses, maxPlotIter){
  let sx = d3.scaleLinear();
  let sy = d3.scaleLinear();

  function updateScales(){
    let width = svg.node().clientWidth;
    let height = svg.node().clientHeight;
    sx.domain([0, maxPlotIter])
    .range([40, width-20]);
    sy.domain(d3.extent(losses))
    .range([height-30,20]);
  }

  

  function draw(){
    svg.selectAll('#lossCurve')
    .data([losses])
    .enter()
    .append('path')
    .attr('id', 'lossCurve');
    let lossCurve = svg.select('#lossCurve');

    lossCurve
    .attr('fill', 'none')
    .attr('stroke', d3.schemeCategory10[0])
    .attr('stroke-width', 1.5)
    .attr('d', d3.line()
      .curve(d3.curveLinear)
      .x((d,i)=>sx(i))
      .y((d)=>sy(d))
    );
  }

  window.addEventListener('resize', ()=>{
    updateScales();
    updateAxes(svg, sx, sy);
    draw();
  });
  updateScales();
  updateAxes(svg, sx, sy);
  draw();
}//traceLoss end


function preprocess(graph, initPos){
  graph.nodes.forEach((d,i)=>{
    d.x = initPos[i][0];
    d.y = initPos[i][1];
  });

  graph.edges.forEach((d,i)=>{
    d.source = graph.nodes.filter(e=>e.id==d.source)[0];
    d.target = graph.nodes.filter(e=>e.id==d.target)[0];
  });
}


function updateNodePosition(graph, xy){
  graph.nodes.forEach((d,i)=>{
    d.x = xy[i][0];
    d.y = xy[i][1];
  });
}


function drawGraph(svg, graph){
  if(svg.sx == undefined){
    svg.sx = d3.scaleLinear();
    svg.sy = d3.scaleLinear();
  }

  function updateScales(){
    let width = svg.node().clientWidth;
    let height = svg.node().clientHeight;

    let xExtent = d3.extent(graph.nodes, d=>d.x);
    let yExtent = d3.extent(graph.nodes, d=>d.y);
    
    // if (svg.xDomain !== undefined){
    //   svg.xDomain[0] = Math.min(xExtent[0], svg.xDomain[0]);
    //   svg.xDomain[1] = Math.max(xExtent[1], svg.xDomain[1]);
    //   svg.yDomain[0] = Math.min(yExtent[0], svg.yDomain[0]);
    //   svg.yDomain[1] = Math.max(yExtent[1], svg.yDomain[1]);
    // }else{
      svg.xDomain = xExtent;
      svg.yDomain = yExtent;
    // }
    
    xExtent = svg.xDomain.slice(0);
    yExtent = svg.yDomain.slice(0);
    let xSize = xExtent[1] - xExtent[0];
    let ySize = yExtent[1] - yExtent[0];

    let xViewport = [30, width-20];
    let yViewport = [height-20,20];
    let drawWidth = xViewport[1] - xViewport[0];
    let drawHeight = yViewport[0] - yViewport[1];

    if (drawWidth/drawHeight > xSize/ySize){
      let adjust = (ySize / drawHeight * drawWidth) - xSize;
      xExtent[0] -= adjust/2;
      xExtent[1] += adjust/2;
    }else{
      let adjust = (xSize / drawWidth * drawHeight) - ySize;
      yExtent[0] -= adjust/2;
      yExtent[1] += adjust/2;
    }
    
    svg.sx.domain(xExtent)
    .range(xViewport);
    svg.sy.domain(yExtent)
    .range(yViewport);
  }
  

  function draw(){
    svg.selectAll('.edge')
    .data(window.graph.edges)
    .exit()
    .remove();
    let edges = svg.selectAll('.edge')
    .data(window.graph.edges)
    .enter()
    .append('line')
    .attr('class', 'edge')
    .attr('fill', 'none')
    .attr('stroke', '#333')
    .attr('stroke-width', 2)
    .attr('opacity', 0.8);
    edges = svg.selectAll('.edge')
    .attr('x1', d=>svg.sx(d.source.x))
    .attr('x2', d=>svg.sx(d.target.x))
    .attr('y1', d=>svg.sy(d.source.y))
    .attr('y2', d=>svg.sy(d.target.y));

    svg.selectAll('.node')
    .data(window.graph.nodes)
    .exit()
    .remove();
    let newNodes = svg.selectAll('.node')
    .data(window.graph.nodes)
    .enter()
    .append('g')
    .attr('class', 'node')
    .call(
      d3.drag()
      .on('drag', (d)=>{
        boundaries = undefined;
        let x = d3.event.sourceEvent.offsetX;
        let y = d3.event.sourceEvent.offsetY;
        let dx = d3.event.dx;
        let dy = d3.event.dy;
        d.x = svg.sx.invert(x);
        d.y = svg.sy.invert(y);
        let newPos = window.graph.nodes.map(d=>[d.x, d.y]);
        dataObj.x.assign(tf.tensor2d(newPos));
        draw();
      })

    );
    ;

    let newCircles = newNodes
    .append('circle')
    .attr('r', 10)
    .attr('fill', d3.schemeCategory10[0]);

    let newTexts = newNodes
    .append('text')
    .style('font-size', 12)
    .style('fill', '#eee')
    .style('text-anchor', 'middle')
    .style('alignment-baseline', 'middle');

    let nodes = svg.selectAll('.node')
    .attr('transform', d=>`translate(${svg.sx(d.x)},${svg.sy(d.y)})`)
    .moveToFront();
    let texts = nodes.selectAll('text')
    .text(d=>d.id);
    let circles = nodes.selectAll('.circles');
  }

  window.addEventListener('resize', ()=>{
    updateScales();
    updateAxes(svg, svg.sx, svg.sy);
    draw();
  });

  updateScales();
  updateAxes(svg, svg.sx, svg.sy);
  draw();
}//drawGraph end


function updateSvgSize(svg_loss, svg_graph){
  let width =  window.innerWidth/12*8 - 50;
  let height_graph =  window.innerHeight/3*2-20;
  let height_loss =  window.innerHeight/3*1-20;
  svg_loss
  .attr('width', '100%')
  .attr('height', height_loss);
  svg_graph
  .attr('width', '100%')
  .attr('height', height_graph);
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

// main
let isPlaying = true;
window.onload = function(){
  let graphTypeSelect = d3.select('#graphType');
  let fn = `data/${graphTypeSelect.node().value}.json`;

  let lrSlider = d3.select('#lr');
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

  let optimizers = [tf.train.momentum(lr, momentum, false)];
  window.optimizers = optimizers;
  let losses = [];
  let animId = 0;

  let svg_loss = d3.select('#loss');
  let svg_graph = d3.select('#graph');

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
        x = tf.variable(
          tf.randomUniform([graph.nodes.length,2], -1, 1)
        );
        boundaries = undefined;
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
          losses.push(record.loss);
          if (losses.length>maxPlotIter){
            losses = losses.slice(losses.length-maxPlotIter);
          }
          traceLoss(svg_loss, losses, maxPlotIter);

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

