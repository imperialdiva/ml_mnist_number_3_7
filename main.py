import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd

    return (pd,)


@app.cell
def _():
    import torch

    return (torch,)


@app.cell
def _():
    from fastai.vision.all import untar_data, URLs, get_image_files, ImageDataLoaders, Image, array, tensor, show_image, F

    return F, Image, URLs, array, show_image, tensor, untar_data


@app.cell
def _(URLs, untar_data):
    path = untar_data(URLs.MNIST_SAMPLE)
    return (path,)


@app.cell
def _(path):
    (path/'train').ls()
    return


@app.cell
def _(path):
    threes = (path/'train'/'3').ls().sorted() 
    sevens = (path/'train'/'7').ls().sorted()
    return sevens, threes


@app.cell
def _(threes):
    threes
    return


@app.cell
def _(sevens):
    sevens
    return


@app.cell
def _(Image, threes):
    im3_path = threes[1]
    im3 = Image.open(im3_path)
    im3
    return (im3,)


@app.cell
def _(array, im3):
    array(im3)[4:10, 4:10]
    return


@app.cell
def _(im3, tensor):
    tensor(im3)[4:10, 4:10]
    return


@app.cell
def _(im3, pd, tensor):
    im3_t = tensor(im3)
    df = pd.DataFrame(im3_t[4:15, 4:22])
    df.style.set_properties(**{'font-size':'6pt'}).background_gradient('Greys')
    return


@app.cell
def _(Image, sevens, tensor, threes):
    from tqdm import tqdm

    seven_tensors = [tensor(Image.open(o)) for o in tqdm(sevens, desc="Загрузка семёрок")]
    three_tensors = [tensor(Image.open(o)) for o in tqdm(threes, desc="Загрузка троек")]
    return seven_tensors, three_tensors


@app.cell
def _(show_image, three_tensors):
    show_image(three_tensors[1])
    return


@app.cell
def _(seven_tensors, three_tensors, torch):
    stacked_seven = torch.stack(seven_tensors).float()/255
    stacked_threes = torch.stack(three_tensors).float()/255
    stacked_threes.shape
    return stacked_seven, stacked_threes


@app.cell
def _(show_image, stacked_threes):
    mean3 = stacked_threes.mean(0)
    show_image(mean3)
    return (mean3,)


@app.cell
def _(show_image, stacked_seven):
    mean7 = stacked_seven.mean(0)
    show_image(mean7)
    return (mean7,)


@app.cell
def _(show_image, stacked_threes):
    a_3 = stacked_threes[1]
    show_image(a_3)
    return (a_3,)


@app.cell
def _(a_3, mean3, mean7):
    dist_3_abc = (a_3 - mean3).abs().mean()
    dist_3_sqr = ((a_3 - mean3)**2).abs().mean().sqrt()
    dist_3_abc, dist_3_sqr

    dist_7_abc = (a_3 - mean7).abs().mean()
    dist_7_sqr = ((a_3 - mean7)**2).abs().mean().sqrt()
    dist_7_abc, dist_7_sqr

    return


@app.cell
def _(F, a_3, mean7):
    F.l1_loss(a_3.float(), mean7), F.mse_loss(a_3.float(), mean7).sqrt()
    return


@app.cell
def _(Image, path, tensor, torch):
    valid_3_tens = torch.stack([tensor(Image.open(o)) for o in (path/'valid'/'3').ls()])
    valid_3_tens = valid_3_tens.float()/255
    valid_7_tens = torch.stack([tensor(Image.open(o)) for o in (path/'valid'/'7').ls()])
    valid_7_tens = valid_7_tens.float()/255
    valid_3_tens.shape, valid_7_tens.shape
    return (valid_3_tens,)


@app.function
def mnist_distance(a, b):
    return (a-b).abs().mean((-1, -2))


@app.cell
def _(a_3, mean3):
    mnist_distance(a_3, mean3)
    return


@app.cell
def _(mean3, valid_3_tens):
    valid_3_dist = mnist_distance(valid_3_tens, mean3)
    valid_3_dist, valid_3_dist.shape
    return


@app.cell
def _(mean3, mean7):
    def is_3(x):
        return mnist_distance(x, mean3) < mnist_distance(x, mean7)

    return (is_3,)


@app.cell
def _(a_3, is_3):
    is_3(a_3), is_3(a_3).float()
    return


if __name__ == "__main__":
    app.run()
