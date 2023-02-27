import bpy

props = {
    "blockType": "block_type",
    "usPoll": "us_poll",
    "msRate": "ms_rate",
    "fmatName": "fmat_name",
    "pmlBorder": "pml_border",
    "snappedName": "snapped_name",
    "s_tstartUnits": "s_tstart_units",
    "s_triseUnits": "s_trise_units",
    "s_durationUnits": "s_duration_units",
    "s_tfallUnits": "s_tfall_units",
    "p_axisSign": "p_axis_sign",
    "p_dispScale": "p_disp_scale",
    "p_pixelRep": "p_pixel_rep",
    "p_imageAlpha": "p_image_alpha",
    "p_magScale": "p_mag_scale",
    "p_dispIsMesh": "p_disp_is_mesh",
    "p_dispIsPlot": "p_disp_is_plot",
    "p_legendLoc": "p_legend_loc",
    "p_plotScale": "p_plot_scale",
    "p_dispColor": "p_disp_color",
    "p_dispPos": "p_disp_pos",
}

for ob in bpy.data.objects:
    for key in ob.keys():
        new_key = props.get(key)
        if new_key:
            ob[new_key] = ob[key]
            del ob[key]
