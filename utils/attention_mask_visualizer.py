import numpy as np
# Print the matrix with words as row labels
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BLACK_SQUARE = "■"
WHITE_SQUARE = "⬚"
def generate_sliding_window_mask_matrix(words, sliding_window=0, img_token="<img>"):
    n = len(words)
    max_word_length = max(len(word) for word in words) 
    first_img_idx = 0

    if sliding_window != 0:
        mask = np.tril(np.zeros((n, n), dtype=int))
        for i in range(n):
            mask[i, max(0, i - sliding_window + 1):i + 1] = 1
    else:
        mask = np.tril(np.ones((n, n), dtype=int))

    for i, k in enumerate(words):
        if img_token in k and not first_img_idx:
            first_img_idx = i
        if first_img_idx > 0 and (img_token not in k or i == n - 1):
            if i == n - 1:
                i += 1
            mask[first_img_idx:i, first_img_idx:i] = 1
            first_img_idx = 0

    vertical_header = [str(idx).rjust(len(str(n))) for idx, word in enumerate(words) ]
    vertical_header = list(map(list, zip(*vertical_header)))  # Transpose
    
    # Print the vertical header
    for row in vertical_header:
        print((max_word_length + 5) * ' ' + " ".join(c for c in row))
    

    
    for i, word in enumerate(words):
        colored_word = f"{YELLOW}{word}{RESET}" if img_token in word else word
        base_display = colored_word.ljust(max_word_length) + ": " + str(i).rjust(len(str(n))) + " "
        row_display = " ".join(
            f"{YELLOW}{BLACK_SQUARE}{RESET}" if img_token in words[j] and mask[i, j] and img_token in words[i] else 
            f"{GREEN}{BLACK_SQUARE}{RESET}" if i == j else 
            BLACK_SQUARE if mask[i, j] else WHITE_SQUARE
            for j in range(n)
        )
        print(base_display + row_display)
    
    print(" " * len(base_display) + "-" * len(row_display))


sentece = "What is this? <img> <img> <img> <img> This is a cat. And these ? <img> <img> <img> <img> <img> These are dogs."
words = sentece.split()
generate_sliding_window_mask_matrix(words)
generate_sliding_window_mask_matrix(words, sliding_window=3)

# Should print:
""""


                              1 1 1 1 1 1 1 1 1 1 2 2
          0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
What :  0 ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
is   :  1 ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
this?:  2 ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
<img>:  3 ■ ■ ■ ■ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
<img>:  4 ■ ■ ■ ■ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
<img>:  5 ■ ■ ■ ■ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
<img>:  6 ■ ■ ■ ■ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
This :  7 ■ ■ ■ ■ ■ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
is   :  8 ■ ■ ■ ■ ■ ■ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
a    :  9 ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
cat. : 10 ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
And  : 11 ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
these: 12 ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
?    : 13 ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
<img>: 14 ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ⬚ ⬚ ⬚
<img>: 15 ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ⬚ ⬚ ⬚
<img>: 16 ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ⬚ ⬚ ⬚
<img>: 17 ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ⬚ ⬚ ⬚
<img>: 18 ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ⬚ ⬚ ⬚
These: 19 ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ⬚ ⬚
are  : 20 ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ⬚
dogs.: 21 ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■
          ----------------------------------------------------
                              1 1 1 1 1 1 1 1 1 1 2 2
          0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
What :  0 ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
is   :  1 ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
this?:  2 ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
<img>:  3 ⬚ ■ ■ ■ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
<img>:  4 ⬚ ⬚ ■ ■ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
<img>:  5 ⬚ ⬚ ⬚ ■ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
<img>:  6 ⬚ ⬚ ⬚ ■ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
This :  7 ⬚ ⬚ ⬚ ⬚ ⬚ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
is   :  8 ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
a    :  9 ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
cat. : 10 ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
And  : 11 ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
these: 12 ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
?    : 13 ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ■ ■ ■ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚
<img>: 14 ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ■ ■ ■ ■ ■ ■ ■ ⬚ ⬚ ⬚
<img>: 15 ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ■ ■ ■ ■ ■ ■ ⬚ ⬚ ⬚
<img>: 16 ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ■ ■ ■ ■ ■ ⬚ ⬚ ⬚
<img>: 17 ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ■ ■ ■ ■ ■ ⬚ ⬚ ⬚
<img>: 18 ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ■ ■ ■ ■ ■ ⬚ ⬚ ⬚
These: 19 ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ■ ■ ■ ⬚ ⬚
are  : 20 ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ■ ■ ■ ⬚
dogs.: 21 ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ⬚ ■ ■ ■
          ----------------------------------------------------

"""