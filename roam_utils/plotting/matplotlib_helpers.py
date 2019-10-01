def add_dim_to_fig_geometry(fig, vertical=True):
    # Now later you get a new subplot; change the geometry of the existing
    n = len(fig.axes)
    for i in range(n):
        if vertical:
            fig.axes[i].change_geometry(n + 1, 1, i + 1)
        else:
            fig.axes[i].change_geometry(1, n+1, i+1)
    return n+1


def get_new_subplot_ax(fig):
    n = add_dim_to_fig_geometry(fig)
    ax = fig.add_subplot(n, 1, n)
    ax.set_autoscale_on(False)
    return ax

#
# def get_new_subplot_ax(fig, sharex=None):
#     n = add_dim_to_fig_geometry(fig)
#     ax = fig.add_subplot(n, 1, n, sharex=sharex)
#     return ax


def set_axis_xy_lim(axis, lim_x, lim_y):
    axis.set_xlim(lim_x[0], lim_x[1])
    axis.set_ylim(lim_y[0], lim_y[1])
    #axis.set_autoscale_on(False)


def set_axis_ticksize(axis, fontsize):
    axis.tick_params(axis='both', which='major', labelsize=fontsize)