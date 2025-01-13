use std::slice::from_raw_parts;
// ============================================================================
// This code is part of RPB.
// ----------------------------------------------------------------------------
// MIT License
//
// Copyright (c) 2023-present Javad Abdi, Mark C. Jeffrey
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// ============================================================================

use rayon::prelude::*;

static _LOG_BLOCK_SIZE: usize = 10;
pub static _BLOCK_SIZE: usize = 1 << _LOG_BLOCK_SIZE;

#[inline]
pub fn num_blocks(n: usize, block_size: usize) -> usize {
    if n == 0 {
        0
    } else {
        1 + (n - 1) / block_size
    }
}

/// reduce array `arr` using a binary associative operator `op` in serial
pub fn reduce_serial<T, F>(arr: &[T], op: F) -> T
where
    T: Default + Copy,
    F: Fn(T, T) -> T,
{
    let mut r = T::default();
    arr.iter().for_each(|x| r = op(r, *x));
    r
}

/// reduce array `arr` using a binary associative operator `op` in parallel
pub fn reduce<T, F>(arr: &[T], op: F) -> T
where
    T: Default + Copy + Send + Sync,
    F: Fn(T, T) -> T + Clone + Send + Sync,
{
    let n = arr.len();
    let block_size = _BLOCK_SIZE.max(4 * (n as f64).sqrt().ceil() as usize);
    let l = num_blocks(n, block_size);
    match l {
        0 => T::default(),
        1 => reduce_serial(arr, op),
        _ => {
            // break into blocks and calculate prefix sum of each block
            let sums: Vec<T> = arr
                .par_chunks(block_size)
                .map(|chunk| reduce_serial(chunk, op.clone()))
                .collect();

            // calculate prefix sum of the block sums
            reduce(&sums, op)
        }
    }
}

/// scan operation on `arr` using a binary associative operator `op` in serial
pub fn scan_serial<T, F>(inp: &[T], out: &mut [T], init: T, inclusive: bool, op: F) -> T
where
    T: Copy + Clone,
    F: Fn(T, T) -> T,
{
    // let mut r = offset;
    // if inclusive {
    //     inp
    //         .iter()
    //         .zip(out.iter_mut())
    //         .for_each(|(x, y)| {
    //             r = op(r, *x);
    //             *y = r;
    //         });
    // } else {
    //     inp
    //         .iter()
    //         .zip(out.iter_mut())
    //         .for_each(|(x, y)| {
    //             let t = *x; // to work when inp == out
    //             *y = r;
    //             r = op(r, t);
    //         });
    // }
    // r
    if inclusive {
        inp.into_iter()
            .zip(out)
            .scan(init, |state, val| {
                *state = op(*state, *val.0);
                *val.1 = *state;
                Some(*state)
            })
            .last()
    } else {
        inp.into_iter()
            .zip(out)
            .scan(init, |state, val| {
                *val.1 = *state;
                *state = op(*state, *val.0);
                Some(*state)
            })
            .last()
    }
    .unwrap_or(init)
}

/// scan operation on `arr` using a binary associative operator `op` in serial
/// this version is inplace and uses the same array for input and output
pub fn scan_serial_inplace<T, F>(inp: &mut [T], init: T, inclusive: bool, op: F) -> T
where
    T: Copy + Clone,
    F: Fn(T, T) -> T,
{
    /* let inp_shadow = unsafe { from_raw_parts(inp.as_ptr(), inp.len()) };
    scan_serial(inp_shadow, inp, offset, inclusive, op) */
    if inclusive {
        inp.into_iter()
            .scan(init, |state, val| {
                *state = op(*state, *val);
                *val = *state;
                Some(*state)
            })
            .last()
    } else {
        inp.into_iter()
            .scan(init, |state, val| {
                let temp = *val;
                *val = *state;
                *state = op(*state, temp);
                Some(*state)
            })
            .last()
    }
    .unwrap_or(init)
}

/// scan operation on `arr` using a binary associative operator `op` in parallel
pub fn scan_<T, F>(inp: &[T], out: &mut [T], inclusive: bool, op: F) -> T
where
    T: Default + Send + Sync + Clone + Copy,
    F: Fn(T, T) -> T + Clone + Send + Sync,
{
    let n = inp.len();
    let l = num_blocks(n, _BLOCK_SIZE);

    // if the array is small, do it sequentially
    if l <= 2 {
        return scan_serial(inp, out, T::default(), inclusive, op);
    }

    // break into blocks and calculate prefix sum of each block
    let mut sums: Vec<T> = Vec::with_capacity(l);
    inp.par_chunks(_BLOCK_SIZE)
        .map(|chunk| chunk.iter().fold(T::default(), |a, b| op(a, *b)))
        .collect_into_vec(&mut sums);

    // perform a scan on block sums to get the each block's offset
    let blk_offset = scan_serial_inplace(&mut sums, T::default(), false, op.clone());

    // scan each block in parallel with its offset
    (
        inp.par_chunks(_BLOCK_SIZE),
        out.par_chunks_mut(_BLOCK_SIZE),
        sums,
    )
        .into_par_iter()
        .for_each(|(in_chunk, l_out, sum)| {
            scan_serial(in_chunk, l_out, sum, inclusive, op.clone());
        });

    blk_offset
}

/// in-place scan operation on `arr`
/// using a binary associative operator `op` in parallel
pub fn scan_inplace<T, F>(inp: &mut [T], inclusive: bool, op: F) -> T
where
    T: Default + Send + Sync + Clone + Copy,
    F: Fn(T, T) -> T + Clone + Send + Sync,
{
    let n = inp.len();
    let l = num_blocks(n, _BLOCK_SIZE);

    // if the array is small, do it sequentially
    if l <= 2 {
        return scan_serial_inplace(inp, T::default(), inclusive, op);
    }

    // break into blocks and calculate prefix sum of each block
    let mut sums: Vec<T> = Vec::with_capacity(l);
    inp.par_chunks(_BLOCK_SIZE)
        .map(|chunk| chunk.iter().fold(T::default(), |a, b| op(a, *b)))
        .collect_into_vec(&mut sums);

    // perform a scan on block sums to get the each block's offset
    let blk_offset = scan_serial_inplace(&mut sums, T::default(), false, op.clone());

    // scan each block in parallel with its offset
    (inp.par_chunks_mut(_BLOCK_SIZE), sums)
        .into_par_iter()
        .for_each(|(in_chunk, sum)| {
            scan_serial_inplace(in_chunk, sum, inclusive, op.clone());
        });

    blk_offset
}

/// Counts the number of elements in `arr` that are true in serial
pub fn sum_bool_serial(arr: &[bool]) -> usize {
    let mut r = 0;
    let n = arr.len();
    let mut i = 0;
    while i < n {
        r += arr[i] as usize;
        i += 1;
    }

    r
}
