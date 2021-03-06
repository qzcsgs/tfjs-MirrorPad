# tfjs-MirrorPad

## Quickstart
``` bash
$ npm install
$ node mirror_pad.js
```

## Details
create square tensor4d
```js
// shape [1, 2, 2, 3]
const x = tf.tensor4d([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]]);
/**
 * x
 * [[
 *   [[1, 2, 3], [4 , 5 , 6 ]],
 *   [[7, 8, 9], [10, 11, 12]]
 * ]]
 */
```

fill right
```js
/**
 * rightIndex
 * [1]
 */
let indices = tf.tensor1d(rightIndex, 'int32');
/**
 * b
 * [[
 *   [[4 , 5 , 6 ]],
 *   [[10, 11, 12]]
 * ]]
 */
let b = x.gather(indices, 2); // axis = 2
/**
 * result
 * [[
 *   [[1, 2, 3], [4 , 5 , 6 ], [4 , 5 , 6 ]],
 *   [[7, 8, 9], [10, 11, 12], [10, 11, 12]]
 * ]]
 */
let result = tf.concat([x, b], 2); // axis = 2
```

```js
/**
 * After the left and right filling is completed
 * [[
 *   [[1, 2, 3],[1, 2, 3], [4 , 5 , 6 ], [4 , 5 , 6 ]],
 *   [[7, 8, 9],[7, 8, 9], [10, 11, 12], [10, 11, 12]]
 * ]]
 */
```

fill top
```js
/**
 * leftTopIndex
 * [0]
 */
indices = tf.tensor1d(leftTopIndex, 'int32');
/**
 * b
 * [[
 *   [[1, 2, 3],[1, 2, 3], [4 , 5 , 6 ], [4 , 5 , 6 ]]
 * ]]
 */
b = result.gather(indices, 1); // axis = 1
/**
 * result
 * [[
 *   [[1, 2, 3],[1, 2, 3], [4 , 5 , 6 ], [4 , 5 , 6 ]],
 *   [[1, 2, 3],[1, 2, 3], [4 , 5 , 6 ], [4 , 5 , 6 ]],
 *   [[7, 8, 9],[7, 8, 9], [10, 11, 12], [10, 11, 12]]
 * ]]
 */
result = tf.concat([b, result], 1); // axis = 1
```

final effect
```js
/**
 * [[
 *   [[1, 2, 3],[1, 2, 3], [4 , 5 , 6 ], [4 , 5 , 6 ]],
 *   [[1, 2, 3],[1, 2, 3], [4 , 5 , 6 ], [4 , 5 , 6 ]],
 *   [[7, 8, 9],[7, 8, 9], [10, 11, 12], [10, 11, 12]]，
 *   [[7, 8, 9],[7, 8, 9], [10, 11, 12], [10, 11, 12]]
 * ]]
 */
```
