// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef __BERT_PATTERN_PARAMS
#define __BERT_PATTERN_PARAMS 1
extern "C" {
    void setSplits(int ffnSplits);
    int getFfnSplits();

    void setVocabSize(int v);
    int getVocabSize();

    void setEmbeddingSize(int v);
    int getEmbeddingSize();

    void setHiddenSize(int v);
    int getHiddenSize();

}
#endif
