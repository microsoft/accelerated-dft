def niceview(view,data):
    view.addVolumetricData(
        data,
        "cube",
        {
            "isoval": 0.05,
            "smoothness": 5,
            "opacity": 0.8,
            "volformat": "cube",
            "color": "blue",
        },
    )
    view.addVolumetricData(
        data,
        "cube",
        {
            "isoval": -0.05,
            "smoothness": 5,
            "opacity": 0.8,
            "volformat": "cube",
            "color": "orange",
        },
    )
    return
