const tf = require('@tensorflow/tfjs-node');

// 3x3
const x = tf.tensor4d([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24], [25, 26, 27]]]]);
// const paddingNum = 1;
const paddingNum = 2;
const width = x.shape[1];

// right gather index
var rightIndex = [];
for (var i = 1; i <= paddingNum; i++) {
  rightIndex.push(width - i);
}

// left and top gather index
var leftTopIndex = [];
for (var i = paddingNum - 1; i >= 0; i--) {
  leftTopIndex.push(i);
}

// right
var indices = tf.tensor1d(rightIndex, 'int32');
var b = x.gather(indices, 2);
var result = tf.concat([x, b], 2);

// left
indices = tf.tensor1d(leftTopIndex, 'int32');
b = x.gather(indices, 2);
result = tf.concat([b, result], 2);

// top
indices = tf.tensor1d(leftTopIndex, 'int32');
b = result.gather(indices, 1);
result = tf.concat([b, result], 1);

// bottom
var bottomIndex = [];
for (var i = 1; i <= paddingNum; i++) {
  bottomIndex.push(width + paddingNum - i);
}
indices = tf.tensor1d(bottomIndex, 'int32');
b = result.gather(indices, 1);
result = tf.concat([result, b], 1);

result.print()
