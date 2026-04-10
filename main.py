import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd

    return (pd,)


@app.cell
def _():
    from fastai.vision.all import untar_data, URLs, get_image_files, ImageDataLoaders, Image, array, tensor

    return Image, URLs, array, tensor, untar_data


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


if __name__ == "__main__":
    app.run()
