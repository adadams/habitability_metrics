import numpy as np
from colorcet import cm
from emulator import construct_emulator_grid, construct_lowerD_grid
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable


def set_rotation_xaxis(ax, rotation_period_limits=[1, 256], buffer=0.1):
    ax.set_xscale("log", base=2)

    ax.set_xlim((1 - buffer) ** np.array([1, -1]) * np.asarray(rotation_period_limits))

    tick_values = np.round(np.log2(rotation_period_limits))
    number_of_ticks = int(np.diff(tick_values).squeeze()) + 1
    ax.set_xticks(np.logspace(*tick_values, base=2, num=number_of_ticks))

    ax.xaxis.set_major_formatter(ScalarFormatter())


def plot_scatterplot(
    dimensions,
    dimension_print_list,
    quantity,
    quantity_plot_names,
    scatter_kwargs,
    filetypes,
):
    number_of_plots = len(dimension_print_list)
    subplot_kwargs = {
        "nrows": 1,
        "ncols": number_of_plots,
        "figsize": (5 * number_of_plots, 4),
    }
    if len(quantity_plot_names) < number_of_plots:
        quantity_plot_name = quantity_plot_names[0]
        subplot_kwargs["sharey"] = True

    fig, axes = plt.subplots(**subplot_kwargs)
    for column_number, (
        dimension_print_name,
        (dimension_name, dimension),
        axis,
    ) in enumerate(zip(dimension_print_list, dimensions.items(), axes.flatten())):
        if dimension_name == "rotation_period":
            axis.scatter(2**dimension, quantity, **scatter_kwargs)
            set_rotation_xaxis(
                axis,
                rotation_period_limits=2
                ** np.array([np.min(dimension), np.max(dimension)]),
            )
        else:
            axis.scatter(dimension, quantity, **scatter_kwargs)

        if subplot_kwargs["sharey"]:
            if column_number == 0:
                axis.set_ylabel(quantity_plot_name)
        else:
            axis.set_ylabel(quantity_plot_names[column_number])

        axis.set_xlabel(dimension_print_name)

    fig.tight_layout()
    for filetype in filetypes:
        plt.savefig(f"{quantity.name}_by-dimension_scatterplot.{filetype}", dpi=150)

    return fig, axes


def plot_noneccentric_emulator_grid(
    emulator,
    emulated_print_name,
    emulated_save_name,
    colormap,
    dimensions,
    grid_spacings,
    subplot_kwargs,
    plot_errors=False,
    fig=None,
    ax=None,
):
    if plot_errors:
        emulator_key = "emulated_error"
        emulated_colormap = cm.gray_r
        emulated_save_name += "_error"
    else:
        emulator_key = "emulated_values"
        emulated_colormap = colormap

    save_within_function = False
    if fig is None and ax is None:
        fig, ax = plt.subplots(**subplot_kwargs)
        save_within_function = True

    emulated_rot_and_obl = construct_emulator_grid(emulator.regressor, grid_spacings)
    emulated_rotation_periods, emulated_obliquities = emulated_rot_and_obl["locations"]

    contour_args = (
        2**emulated_rotation_periods,
        emulated_obliquities,
        emulated_rot_and_obl[emulator_key],
    )
    contour_kwargs = dict(levels=10)

    color_contours = ax.contourf(
        *contour_args, **contour_kwargs, cmap=emulated_colormap
    )
    line_contours = ax.contour(*contour_args, **contour_kwargs, cmap=cm.gray)
    ax.clabel(line_contours, line_contours.levels, inline=True, fontsize=16)

    set_rotation_xaxis(
        ax,
        rotation_period_limits=[
            2 ** np.min(dimensions["rotation_period"]),
            2 ** np.max(dimensions["rotation_period"]),
        ],
        buffer=0.0,
    )

    ax.set_xlabel("Rotation Period (days)", fontsize=28)
    ax.set_ylabel(r"Obliquity $\left(^\circ\right)$", fontsize=28)
    ax.tick_params(axis="both", which="major", labelsize=24)

    divider = make_axes_locatable(ax)
    colorbar_ax = divider.append_axes("top", size="10%", pad=0.025)
    colorbar = fig.colorbar(
        color_contours, cax=colorbar_ax, orientation="horizontal", pad=0.05
    )
    colorbar.set_label(label=emulated_print_name, size=32)
    colorbar_ax.xaxis.set_label_coords(0.5, 2.5)
    colorbar_ax.xaxis.set_label_position("top")
    colorbar_ax.xaxis.set_ticks_position("top")
    colorbar_ax.tick_params(axis="x", which="major", labelsize=24)
    colorbar_ax.invert_xaxis()

    fig.tight_layout()

    if save_within_function:
        plt.savefig(
            f"{emulated_save_name}_emulated_grid.pdf", bbox_inches="tight", dpi=300
        )

    return fig, ax


def plot_grid_of_grids(
    emulator,
    actual_values,
    emulated_print_name,
    emulated_save_name,
    colormap,
    dimensions,
    plotted_dimension_names,
    plotted_print_labels,
    fixed_dimension_names,
    fixed_print_labels,
    grid_spacings,
    subplot_kwargs,
    contour_kwargs=dict(),
    scatter_kwargs=dict(),
    plot_errors=False,
):
    fixed_dimensions = {
        name: None for name in plotted_dimension_names + fixed_dimension_names
    }
    outer_x_name, outer_y_name = fixed_dimension_names
    inner_x_name, inner_y_name = plotted_dimension_names

    number_of_rows = subplot_kwargs["nrows"]
    number_of_columns = subplot_kwargs["ncols"]

    if plot_errors:
        emulator_key = "emulated_error"
        emulated_colormap = cm.gray_r
        emulated_save_name += "_error"
    else:
        emulator_key = "emulated_values"
        emulated_colormap = colormap

    fig, axes = plt.subplots(**subplot_kwargs)
    for row_number, (row, fixed_y) in enumerate(
        zip(
            axes,
            np.linspace(
                np.max(grid_spacings[fixed_dimension_names[-1]]),
                np.min(grid_spacings[fixed_dimension_names[-1]]),
                num=number_of_rows,
            ),
        )
    ):
        row_ax = fig.add_subplot(number_of_columns, 1, row_number + 1)
        row_ax.set_ylabel(r"${:g}$".format(fixed_y), fontsize=23)
        row_ax.yaxis.tick_right()
        if row_number == number_of_rows - 1:
            row_ax.set_ylabel(
                fixed_print_labels[-1][:-1] + "=" + row_ax.get_ylabel()[1:]
            )
        row_ax.yaxis.set_label_coords(1.035, 0.5)
        row_ax.xaxis.set_visible(False)
        plt.setp(row_ax.spines.values(), visible=False)
        row_ax.tick_params(right=False, labelright=False)
        row_ax.patch.set_visible(False)

        fixed_dimensions[fixed_dimension_names[-1]] = fixed_y

        for column_number, (cell, fixed_x) in enumerate(
            zip(
                row,
                np.linspace(
                    np.min(grid_spacings[outer_x_name]),
                    np.max(grid_spacings[outer_x_name]),
                    num=number_of_columns,
                ),
            )
        ):
            if row_number == 0:
                column_ax = fig.add_subplot(1, number_of_rows, column_number + 1)
                column_ax.set_xlabel(r"${:g}$".format(fixed_x), fontsize=23)
                column_ax.xaxis.tick_top()
                if column_number == 0:
                    column_ax.set_xlabel(
                        r"$e \cos\phi_\mathrm{peri}=" + column_ax.get_xlabel()[1:]
                    )
                column_ax.xaxis.set_label_coords(0.5, 1.035)
                column_ax.yaxis.set_visible(False)
                plt.setp(column_ax.spines.values(), visible=False)
                column_ax.tick_params(top=False, labeltop=False)
                column_ax.patch.set_visible(False)

            if (column_number == 0) and (row_number == number_of_rows - 1):
                cb_left, cb_bottom, cb_width, cb_height = cell._position.bounds

            if np.sqrt(fixed_x**2 + fixed_y**2) > 0.225:
                if row_number == 0:
                    cell.xaxis.set_visible(False)
                    cell.yaxis.set_visible(False)
                    cell.spines["top"].set_visible(False)
                    cell.spines["bottom"].set_visible(False)
                    cell.spines["left"].set_visible(False)
                    cell.spines["right"].set_visible(False)

                else:
                    cell.set_axis_off()
                continue

            fixed_dimensions[outer_x_name] = fixed_x

            emulated_rot_and_obl = construct_lowerD_grid(
                emulator.regressor,
                grid_spacings,
                plotted_dimension_names,
                fixed_dimensions,
            )

            color_contours = cell.contourf(
                emulated_rot_and_obl["locations"][
                    list(dimensions.keys()).index(inner_x_name)
                ],
                emulated_rot_and_obl["locations"][
                    list(dimensions.keys()).index(inner_y_name)
                ],
                emulated_rot_and_obl[emulator_key],
                cmap=emulated_colormap,
                **contour_kwargs,
            )

            line_contours = cell.contour(
                emulated_rot_and_obl["locations"][
                    list(dimensions.keys()).index(inner_x_name)
                ],
                emulated_rot_and_obl["locations"][
                    list(dimensions.keys()).index(inner_y_name)
                ],
                emulated_rot_and_obl[emulator_key],
                cmap=cm.gray,
                **contour_kwargs,
            )

            def fmt(x):
                s = f"{x:.1f}"
                if s.endswith("0"):
                    s = f"{x:.0f}"
                return rf"{s}" if plt.rcParams["text.usetex"] else f"{s}"

            cell.clabel(line_contours, line_contours.levels, inline=True, fontsize=12)

            training_point_distances = (
                0.2
                + (
                    (dimensions[outer_x_name] - fixed_x)
                    / np.ptp(dimensions[outer_x_name])
                )
                ** 2
                + (
                    (dimensions[fixed_dimension_names[-1]] - fixed_y)
                    / np.ptp(dimensions[fixed_dimension_names[-1]])
                )
                ** 2
            )

            cell.scatter(
                dimensions[inner_x_name],
                dimensions[inner_y_name],
                c=actual_values,
                cmap=colormap,
                s=40 / (2 * training_point_distances),
                edgecolor="#444444",
                zorder=5,
                **scatter_kwargs,
            )

            cell.yaxis.set_major_formatter(FormatStrFormatter("%g"))
            obliquity_tick_positions = [0, 22.5, 45, 67.5, 90]
            obliquity_tick_labels = [
                fmt(position) for position in obliquity_tick_positions
            ]
            if row_number >= 3:
                cell.set_yticks(
                    obliquity_tick_positions[:-1], obliquity_tick_labels[:-1], size=20
                )
            else:
                cell.set_yticks(
                    obliquity_tick_positions, obliquity_tick_labels, size=20
                )

            if np.abs(column_number - 2) != (4 - row_number):
                cell.set_xlabel(None)
                cell.xaxis.set_visible(False)
            if np.abs(row_number - 2) != column_number:
                cell.set_ylabel(None)
                cell.yaxis.set_visible(False)

            if (column_number == (number_of_rows - 1) // 2) and (
                row_number == number_of_rows - 1
            ):
                cell.set_xlabel(plotted_print_labels[0], size=28)
                if inner_x_name == "rotation_period":
                    power_locs = np.arange(0, 10, 2)
                    power_labels = 2**power_locs
                    cell.set_xticks(power_locs, power_labels, size=20, rotation=45)

            if (column_number == 0) and (row_number == (number_of_columns - 1) // 2):
                if inner_x_name == "rotation_period":
                    power_locs = np.arange(0, 10, 2)
                    power_labels = 2**power_locs
                    cell.set_xticks(power_locs, power_labels, size=20, rotation=45)
                cell.set_ylabel(plotted_print_labels[1], size=28)

            elif inner_x_name == "rotation_period":
                power_locs = np.arange(0, 10, 2)
                power_labels = 2**power_locs
                cell.set_xticks(power_locs, power_labels, size=20, rotation=45)

    colorbar_ax = fig.add_axes(
        [
            cb_left - 0.25 * cb_width,
            cb_bottom + 0.33 * cb_height,
            2 * cb_width,
            0.3 * cb_height,
        ]
    )
    colorbar_ax.tick_params(axis="x", which="major", labelsize=24, rotation=45)
    colorbar = fig.colorbar(color_contours, cax=colorbar_ax, orientation="horizontal")
    if plot_errors:
        colorbar.set_label(label="Uncertainty in " + emulated_print_name, size=32)
    else:
        colorbar.set_label(label=emulated_print_name, size=32)
    colorbar_ax.xaxis.set_label_coords(0.5, -1.5)

    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    plt.savefig(
        f"{emulated_save_name}_{inner_x_name}_vs_{inner_y_name}_grid_in_{outer_x_name}_and_{outer_y_name}.pdf",
        bbox_inches="tight",
        dpi=300,
    )

    return fig, axes
