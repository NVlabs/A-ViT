import torch

def filter_probs(probs, thres=0.002,):
    # this returns a final tensor that stops tensor if kl divergence change
    # is below the threshold
    # input - probs lst
    batch_size, _ = probs[0].shape

    out_lst = []
    layer_lst = []

    for idx in range(batch_size):
        out_logits = None
        layer_cnt = 1

        for layer in range(1, 12):
            delta = torch.nn.functional.kl_div(probs[layer][idx,:], probs[layer-1][idx,:], reduction="mean")
            # delta = torch.norm(probs[layer][idx,:] - probs[layer-1][idx,:])

            if (out_logits is None) or abs(delta.data.item()) >= thres:
                out_logits = probs[layer][idx,:]
                layer_cnt = layer + 1
            else:
                break

        out_lst.append(out_logits)
        layer_lst.append(layer_cnt)

    out_tensor = torch.stack(out_lst)

    return out_tensor, layer_lst

def filter_conf(probs, thres=0.002):
    # this returns a final tensor that stops tensor if kl divergence change
    # is below the threshold
    # input - probs lst
    batch_size, _ = probs[0].shape

    out_lst = []
    layer_lst = []

    for idx in range(batch_size):
        out_logits = None
        layer_cnt = 1

        for layer in range(12):
            delta = torch.max(probs[layer][idx,:])
            if (out_logits is None) or abs(delta.data.item()) <= thres:
                out_logits = probs[layer][idx,:]
                layer_cnt = layer + 1
            else:
                break

        out_lst.append(out_logits)
        layer_lst.append(layer_cnt)

    out_tensor = torch.stack(out_lst)

    return out_tensor, layer_lst
