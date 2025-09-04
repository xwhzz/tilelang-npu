lib_path = "./kernel_lib.so"

import torch

SEQ_LEN = 16384

DIM = 128

torch.manual_seed(100)

Q = torch.randn(SEQ_LEN, DIM).half().npu()
K = torch.randn(SEQ_LEN, DIM).half().npu()
V = torch.randn(SEQ_LEN, DIM).half().npu()

Output = torch.zeros(SEQ_LEN, DIM).float().npu()

workspace_1 = torch.zeros(128 * DIM).float().npu()
workspace_2 = torch.zeros(128 * DIM).half().npu()
workspace_3 = torch.zeros(128 * DIM).float().npu()

logsum = torch.zeros(DIM).float().npu()

print("init successful!")

import ctypes

lib = ctypes.CDLL(lib_path)

stream = torch.npu.current_stream()._as_parameter_

def tl():
    return lib.call(
        # noqa: E501
        ctypes.c_void_p(Q.data_ptr()),
        ctypes.c_void_p(K.data_ptr()),
        ctypes.c_void_p(V.data_ptr()),
        ctypes.c_void_p(Output.data_ptr()),
        ctypes.c_void_p(workspace_1.data_ptr()),
        ctypes.c_void_p(workspace_2.data_ptr()),
        ctypes.c_void_p(workspace_3.data_ptr()),
        ctypes.c_void_p(logsum.data_ptr()),
        stream)




tl()

# print(Output)

# print(workspace_1)

# print(workspace_2)

# print(workspace_3)

# print(logsum)


# for i in range(128):
#     Output[i] /= logsum[i]

print(Output)

ref_output = torch.nn.functional.softmax(Q @ K.T / (128**0.5), dim=-1) @ V

print(ref_output)


# print(workspace_1.reshape(SEQ_LEN, SEQ_LEN))

# # print(workspace_2)

# # ref_output = Q @ K.T

# ref_output = torch.zeros(SEQ_LEN, SEQ_LEN).half().npu()


# for i in range(0, SEQ_LEN, 128):
#     for j in range(0, SEQ_LEN, 128):
#         ref_output[i:i+128, j:j+128] = Q[i:i+128, :] @ K[j:j+128, :].T



# print(ref_output)

# torch.testing.assert_close(workspace_1.reshape(SEQ_LEN, SEQ_LEN), ref_output.float(), atol=1e-2, rtol=1e-2)

# torch.testing.assert_close(Output[0 : 16384 : 128, :], ref_output[0 : 16384 : 128, :], atol=1e-2, rtol=1e-2)