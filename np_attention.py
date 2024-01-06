import numpy as np

def soft_max(z):
    t = np.exp(z)
    a = np.exp(z) / np.expand_dims(np.sum(t, axis=-1), -1)
    return a

# attention demo
print("attention demo")
Query = np.array([
    [1,0,2],
    [2,2,2],
    [2,1,3]
])

Key = np.array([
    [0,1,1],
    [4,4,0],
    [2,3,1]
])

Value = np.array([
    [1,2,3],
    [2,8,0],
    [2,6,3]
])

scores = Query @ Key.T
print(scores)
scores = soft_max(scores)
print(scores)
out = scores @ Value
print(out)

# attention for Encoder
print("attention for Encoder")
values_length = 3
num_attention_heads = 8
hidden_size = 768
attention_head_size = hidden_size // num_attention_heads

Query = np.random.rand(values_length, hidden_size)
Key = np.random.rand(values_length, hidden_size)
Value = np.random.rand(values_length, hidden_size)

Query = np.reshape(Query, [values_length, num_attention_heads, attention_head_size])
Key = np.reshape(Key, [values_length, num_attention_heads, attention_head_size])
Value = np.reshape(Value, [values_length, num_attention_heads, attention_head_size])

Query = np.transpose(Query, [1, 0, 2])
Key = np.transpose(Key, [1, 0, 2])
Value = np.transpose(Value, [1, 0, 2])

scores = Query @ np.transpose(Key, [0, 2, 1])
print(np.shape(scores))
scores = soft_max(scores)
print(np.shape(scores))
out = scores @ Value
print(np.shape(out))
out = np.transpose(out, [1, 0, 2])
print(np.shape(out))
out = np.reshape(out, [values_length , 768])
print(np.shape(out))

# attention for Decoder
print("attention for Decoder")
values_length_q = 3
values_length_kv = 6
num_attention_heads = 8
hidden_size = 768
attention_head_size = hidden_size // num_attention_heads

Query = np.random.rand(values_length_q, hidden_size)
Key = np.random.rand(values_length_kv, hidden_size)
Value = np.random.rand(values_length_kv, hidden_size)

Query = np.reshape(Query, [values_length_q, num_attention_heads, attention_head_size])
Key = np.reshape(Key, [values_length_kv, num_attention_heads, attention_head_size])
Value = np.reshape(Value, [values_length_kv, num_attention_heads, attention_head_size])

Query = np.transpose(Query, [1, 0, 2])
Key = np.transpose(Key, [1, 0, 2])
Value = np.transpose(Value, [1, 0, 2])

scores = Query @ np.transpose(Key, [0, 2, 1])
print(np.shape(scores))
scores = soft_max(scores)
print(np.shape(scores))
out = scores @ Value
print(np.shape(out))
out = np.transpose(out, [1, 0, 2])
print(np.shape(out))
out = np.reshape(out, [values_length_q, 768])
print(np.shape(out))