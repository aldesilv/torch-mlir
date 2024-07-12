import torch

def single_ngram_run(skip_count, input_sequence, start_input_idx,
                     pool_int64s, start, ngram_length):
  input_idx = start_input_idx
  pool_idx = start

  matched = 0

  while input_idx < len(input_sequence):
    if pool_int64s[pool_idx] == input_sequence[input_idx]:
      matched += 1
      pool_idx += 1
      input_idx += (skip_count + 1)
    else:
      return 0
    if matched == ngram_length:
      return 1
  return 0

def single_ngram_run_torch(skip_count, input_sequence, start_input_idx,
                           pool_int64s, start, ngram_length):
  pool_ngram = torch.as_strided(pool_int64s, (ngram_length,), (1,), start)
  input_ngram = torch.as_strided(input_sequence, (ngram_length,), (skip_count + 1,), start_input_idx)
  return ( 1 if torch.equal(pool_ngram, input_ngram) else 0)

print(single_ngram_run_torch(0, torch.tensor([1, 3, 3, 7, 7]), 0, \
                             torch.tensor([1, 3, 7]), 0, 2))
print(single_ngram_run(0, [1, 3, 3, 7, 7], 0, \
                       [1, 3, 7], 0, 2))

def count_single_ngram(skip_count, input_sequence,
                       pool_int64s, start, ngram_length):
  count = 0
  for start_input_idx in range(len(input_sequence)):
    count += single_ngram_run(skip_count, input_sequence, start_input_idx,
                              pool_int64s, start, ngram_length)
  return count

print(count_single_ngram(0, [1, 3, 3, 7, 7], \
                         [1, 3, 7], 0, 2))

print(count_single_ngram(0, [1, 3, 3, 7, 7], \
                         [3, 7], 0, 2))

print(count_single_ngram(1, [1, 3, 3, 7, 7], \
                         [3, 7], 0, 2))

print(count_single_ngram(0, [1, 3, 3, 7, 7], \
                         [3, 7], 0, 1))

def count_single_ngram_multiple_skips(max_skip_count, input_sequence,
                                      pool_int64s, start, ngram_length):
  count = 0
  if ngram_length == 1:
    count += count_single_ngram(0, input_sequence,
                                pool_int64s, start, ngram_length)
  else:
    for skip_count in range(max_skip_count + 1):
      count += count_single_ngram(skip_count, input_sequence,
                                  pool_int64s, start, ngram_length)

  return count

print(count_single_ngram_multiple_skips(1, [1, 3, 3, 7, 7], \
                                        [3, 7], 0, 2))

print(count_single_ngram_multiple_skips(1, [1, 3, 3, 7, 7], \
                                        [3, 7], 0, 1))

print(count_single_ngram_multiple_skips(1, [1, 3, 3, 7, 7], \
                                        [1, 3, 7], 0, 3))

def ngram_tf(max_skip_count, input_sequence,
             pool_int64s, ngram_counts, min_gram_length, max_gram_length,
             ngram_indexes, output, is_2d_output, batch=0):
  ngram_i = 0
  for i in range(len(ngram_counts)):
    start_idx = ngram_counts[i]
    end_idx = (
               ngram_counts[i + 1]
               if (i + 1) < len(ngram_counts)
               else len(pool_int64s)
              )
    for start in range(start_idx, end_idx, i+1):
      freq = count_single_ngram_multiple_skips(max_skip_count, input_sequence,
                                               pool_int64s, start, i+1)
      if i+1 >= min_gram_length and i+1 <= max_gram_length:
        if not is_2d_output:
          output[ngram_indexes[ngram_i]] = freq
        else:
          output[batch][ngram_indexes[ngram_i]] = freq
      ngram_i += 1

# tf_uniandbigrams_skip5
output = [0, 0, 0, 0, 0, 0, 0]
ngram_tf(5, [1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8],
         [2, 3, 5, 4, 5, 6, 7, 8, 6, 7], [0, 4], 1, 2,
         [0, 1, 2, 3, 4, 5, 6], output, False)
print(output)

# tf_onlybigrams_skip5
output = [0, 0, 0, 0, 0, 0, 0]
ngram_tf(5, [1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8],
         [2, 3, 5, 4, 5, 6, 7, 8, 6, 7], [0, 4], 2, 2,
         [0, 1, 2, 3, 4, 5, 6], output, False)
print(output)

# tf_onlybigrams_levelempty
output = [0, 0, 0]
ngram_tf(0, [1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8],
         [5, 6, 7, 8, 6, 7], [0, 0], 2, 2,
         [0, 1, 2], output, False)
print(output)

#tf_only_bigrams_skip0
output = [0, 0, 0, 0, 0, 0, 0]
ngram_tf(0, [1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8],
         [2, 3, 5, 4, 5, 6, 7, 8, 6, 7], [0, 4], 2, 2,
         [0, 1, 2, 3, 4, 5, 6], output, False)
print(output)

def batch_ngram_tf(max_skip_count, batch_input,
                   pool_int64s, ngram_counts, min_gram_length, max_gram_length,
                   ngram_indexes, output):
  for i in range(len(batch_input)):
    ngram_tf(max_skip_count, batch_input[i],
             pool_int64s, ngram_counts, min_gram_length, max_gram_length,
             ngram_indexes, output, True, i)

# tf_batch_onlybigrams_skip0
output = [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]
batch_ngram_tf(0, [[1, 1, 3, 3, 3, 7], [8, 6, 7, 5, 6, 8]],
               [2, 3, 5, 4, 5, 6, 7, 8, 6, 7], [0, 4], 2, 2,
               [0, 1, 2, 3, 4, 5, 6], output)
print(output)

# tf_batch_onlybigrams_skip5
output = [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]
batch_ngram_tf(5, [[1, 1, 3, 3, 3, 7], [8, 6, 7, 5, 6, 8]],
               [2, 3, 5, 4, 5, 6, 7, 8, 6, 7], [0, 4], 2, 2,
               [0, 1, 2, 3, 4, 5, 6], output)
print(output)

# tf_batch_uniandbigrams_skip5
output = [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]
batch_ngram_tf(5, [[1, 1, 3, 3, 3, 7], [8, 6, 7, 5, 6, 8]],
               [2, 3, 5, 4, 5, 6, 7, 8, 6, 7], [0, 4], 1, 2,
               [0, 1, 2, 3, 4, 5, 6], output)
print(output)


