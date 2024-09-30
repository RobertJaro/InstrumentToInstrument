from imreg_dft import transform_img_dict, similarity


def register(d_1, d_2, missing_val=0, strides=1,
             constraints={'scale': (1, 0), 'angle': (0, 10), 'tx': (0, 0), 'ty': (0, 0)}):
    transformation = similarity(d_2[::strides, ::strides], d_1[::strides, ::strides], numiter=20,
                                constraints=constraints)

    transformation['tvec'] = (transformation['tvec'][0] * strides, transformation['tvec'][1] * strides)
    print(transformation['tvec'], transformation['angle'])
    d_1_registered = transform_img_dict(d_1, transformation, bgval=missing_val, order=3)
    return d_1_registered
