import kine_cart2D
import matplotlib.pyplot as plt

def contour_plot(xMesh, yMesh, grad, saveName, ax=None, figsize=(3.5, 3.5)):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    surf = ax.contourf(xMesh, yMesh, grad, cmap='jet')
    plt.savefig("./pics/" + saveName, dpi=600)
    plt.close(fig)

if __name__ == "__main__":
    data_path = './Sarah_dat/datasets/Taylor_2021_2022/COsi_v4/'
    dataList = [
        data_path + '2019_2020_bef_bottom.txt',
        data_path + '2019_2020_bef_lowandno.txt',
        data_path + '2019_2020_bef_outside.txt',
        data_path + '2019_2020_bef_top.txt'
    ]

    # dataList = [
    #     data_path + '20191_2020_bottom.txt',
    #     data_path + '20191_2020_lowandno.txt',
    #     data_path + '20191_2020_outside.txt',
    #     data_path + '20191_2020_top.txt'
    # ]

    df = kine_cart2D.dat_combine(dataList)
    xMesh, yMesh =  kine_cart2D.gen_mesh(df)
    xDisp, yDisp =  kine_cart2D.gen_disp(df, xMesh, yMesh)
    kinematics = kine_cart2D.Kine_2d(xMesh, yMesh, xDisp, yDisp)
    div = kinematics.divergence()
    curl = kinematics.curl()

    # contour_plot(xMesh, yMesh, div, 'cart2D_div_4m.jpg')
    # contour_plot(xMesh, yMesh, curl, 'cart2D_curl_4m.jpg')
    contour_plot(xMesh, yMesh, div, 'cart2D_div_test.jpg')
    contour_plot(xMesh, yMesh, curl, 'cart2D_curl_test.jpg')


