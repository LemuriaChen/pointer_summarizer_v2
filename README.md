
### 自定义的类
 
#### Example 类

* 作用：对单个实例，包含输入(article)和输出(abstract)的一个包装类

* 包含对象：
    * enc_len：max(article中token的个数, max_enc_steps)
    * enc_input：article中token对应字典中的索引
    * dec_input：abstract中token对应字典中的索引
    * target：abstract中token对应字典中的索引（相对dec_input往后偏移一格）
    * dec_len：max(abstract中token的个数, max_dec_steps)
    * enc_input_extend_vocab：article中token对应字典中的索引（如果有oov，则增加扩展词典，词典的维度增加unique(oov)的个数），这样的话enc_input_extend_vocab中不会再出现UNK对应的索引
    * article_oov：article中oov出现的列表（列表中无重复元素）
    * original_article：article中原始的字符串
    * original_abstract：abstract中原始的字符串
    * original_abstract_sentences：abstract中原始的字符串转化为的句子字符串的列表

* 包含方法：
    * get_dec_inp_tgt_seqs：给定abstract单词对应的索引列表，得到截断的decoder的输入和输出
    * pad_decoder_inp_tgt：给解码器填充PAD_token id
    * pad_encoder_input：给编码器填充PAD_token id

#### Batch

* 作用：对 Example 类 的一个包装类，给定输入是 Example 类的一个列表 
    
* 包含对象：
    * batch_size：批处理样本数
    * pad_id：PAD_token id
    * enc_batch：编码器输入 enc_input 矩阵
    * enc_lens：编码器长度 enc_len 列表
    * enc_padding_mask：编码器非 PAD_token 零一矩阵
    * max_art_oov：编码器输入 最大oov数，标量
    * art_oov：编码器oov，列表的列表
    * enc_batch_extend_vocab：同enc_batch，加入了oov编码的 enc_input矩阵
    * dec_batch：解码器输入矩阵
    * target_batch：解码器目标矩阵
    * dec_padding_mask：解码器非 PAD_token 零一矩阵
    * dec_lens：解码器长度列表
    * original_articles：原始文章字符串列表
    * original_abstracts：原始摘要字符串列表
    * original_abstracts_sentences：原始摘要字符串列表
    

#### Batcher

* 作用：对 Batch 类的一个包装类

* 包含对象：


