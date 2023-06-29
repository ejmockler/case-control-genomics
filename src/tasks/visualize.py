import os
import traceback
from types import SimpleNamespace
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import neptune
import numpy as np
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    auc,
    confusion_matrix,
)
from skopt.plots import plot_convergence
import shap
import gc
from config import config

matplotlib.use("agg")


def plotCalibration(title, labelsPredictionsByInstance, config):
    # code from https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html
    fig, ax_calibration_curve = plt.subplots(figsize=(10, 10))
    colors = plt.cm.get_cmap("Dark2")

    calibration_displays = {}
    for i, (name, (labels, predictions)) in enumerate(
        labelsPredictionsByInstance.items()
    ):
        display = CalibrationDisplay.from_predictions(
            [
                config["clinicalTable"]["caseAlias"] if label == 1 else label
                for label in labels
            ],
            predictions,
            pos_label=config["clinicalTable"]["caseAlias"],
            n_bins=10,
            name=name,
            ax=ax_calibration_curve,
            color=colors(i),
        )
        calibration_displays[name] = display

    ax_calibration_curve.grid()
    title = "\n".join(line.strip() for line in title.split("\n"))
    ax_calibration_curve.set_title(title)

    # Add histogram
    # grid_positions = [(i+2,j) for i in range(len(predictionsByModelName.keys())//2) for j in range(2)]
    # for i, modelName in enumerate(predictionsByModelName.keys()):
    #     row, col = grid_positions[i]
    #     ax = fig.add_subplot(gs[row, col])
    #     ax.hist(
    #         calibration_displays[modelName].y_prob,
    #         range=(0, 1),
    #         bins=10,
    #         label=modelName,
    #         color=colors(i),
    #     )
    #     ax.set(title=modelName, xlabel="Mean predicted probability", ylabel="Count")

    plt.tight_layout()
    return fig


def plotAUC(
    title, labelsPredictionsByInstance=None, tprFprAucByInstance=None, config=config
):
    # trace AUC for each set of predictions
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(10, 10))
    if labelsPredictionsByInstance is not None:
        for name, (labels, predictions) in labelsPredictionsByInstance.items():
            # plot ROC curve for this fold
            viz = RocCurveDisplay.from_predictions(
                [
                    config["clinicalTable"]["caseAlias"]
                    if label == 1
                    else config["clinicalTable"]["controlAlias"]
                    for label in labels
                ],
                predictions,
                name=name,
                pos_label=config["clinicalTable"]["caseAlias"],
                alpha=0.6,
                lw=2,
                ax=ax,
            )
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)
    elif tprFprAucByInstance is not None:
        for name, (tprList, fprList, aucScore) in tprFprAucByInstance.items():
            viz = RocCurveDisplay(
                tpr=tprList,
                fpr=fprList,
                roc_auc=aucScore,
                estimator_name=name,
            )
            viz.plot(
                alpha=0.6,
                lw=2,
                ax=ax,
            )
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)

    # summarize ROCs per fold and plot standard deviation
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=4,
        alpha=0.8,
    )
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    title = "\n".join(line.strip() for line in title.split("\n"))
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title=title)
    ax.legend(loc="lower right")
    ax.set(title=title)
    plt.tight_layout()
    return fig


def plotConfusionMatrix(title, labelsPredictionsByInstance, config):
    all_labels = []
    all_predictions = []
    matrix_figures = []

    for name, (labels, predictions) in labelsPredictionsByInstance.items():
        # Ensure any probabilities become predictions
        predictions = np.around(predictions).astype(int)
        # Compute confusion matrix for this fold
        matrix = confusion_matrix(labels, predictions)

        # Create ConfusionMatrixDisplay for this fold
        disp = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=[
                config["clinicalTable"]["controlAlias"],
                config["clinicalTable"]["caseAlias"],
            ],
        )

        fig, ax = plt.subplots()
        disp.plot(
            include_values=True, cmap="viridis", ax=ax, xticks_rotation="horizontal"
        )

        title = "\n".join("    " + line.strip() for line in title.split("\n"))
        ax.set_title(title, fontsize=8)

        plt.tight_layout()
        matrix_figures.append(fig)
        plt.close(fig)

        # Collect all labels and predictions for later use
        all_labels.extend(labels)
        all_predictions.extend(predictions)

    avg_matrix = confusion_matrix(all_labels, all_predictions, normalize="all")

    # Create ConfusionMatrixDisplay for the average confusion matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=avg_matrix,
        display_labels=[
            config["clinicalTable"]["controlAlias"],
            config["clinicalTable"]["caseAlias"],
        ],
    )

    avgFig, ax = plt.subplots()
    cm_display = disp.plot(
        include_values=True, cmap="viridis", ax=ax, xticks_rotation="horizontal"
    )

    # List normalized proportions as percentages
    for i in range(avg_matrix.shape[0]):
        for j in range(avg_matrix.shape[1]):
            percentage = avg_matrix[i, j] * 100
            cm_display.text_[i, j].set_text(f"{percentage:.1f}%")  # 1 decimal place
            cm_display.text_[i, j].set_fontsize(7)

    title = "\n".join(
        "    " + line.strip()
        for line in f"""
        {title}
        Average across folds
        """.split(
            "\n"
        )
    )

    ax.set_title(title, fontsize=8)
    ax.set_xlabel("Predicted label", fontsize=8)
    ax.set_ylabel("True label", fontsize=8)
    ax.tick_params(axis="both", which="major", labelsize=7)

    colorbar = cm_display.im_.colorbar
    colorbar.ax.tick_params(labelsize=7)

    # Convert colorbar ticks to percentages
    tick_vals = colorbar.get_ticks()
    tick_labels = ["{:.0f}%".format(val * 100) for val in tick_vals]
    colorbar.set_ticklabels(tick_labels)
    colorbar.ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.close("all")

    return matrix_figures, avgFig


def plotSampleAccuracy():
    pass


def plotOptimizer(title, resultsByInstance):
    # code from https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(2, 2)
    colors = plt.cm.get_cmap("Dark2")
    ax_convergence = fig.add_subplot(gs[:2, :2])
    plot_convergence(
        *[(modelName, result) for modelName, result in resultsByInstance.items()],
        ax=ax_convergence,
        color=colors,
    )
    title = "\n".join(line.strip() for line in title.split("\n"))
    ax_convergence.set(title=title)
    plt.tight_layout()
    return fig


def plotSample(
    j, k, runID, modelName, plotSubtitle, current, holdout=False, config=config
):
    import matplotlib.pyplot as plt
    import shap

    currentLabel = (
        current["testLabels"][j][k] if not holdout else current["holdoutLabels"][j][k]
    )
    sampleID = current["testIDs"][j][k] if not holdout else current["holdoutIDs"][j][k]
    localExplanations = (
        current["localExplanations"][j]
        if not holdout
        else current["holdoutLocalExplanations"][j]
    )
    waterfallPlot = plt.figure()
    title = "\n".join(
        line.strip()
        for line in f"""
            {sampleID}
            Shapely explanations from {modelName}
            Fold {j+1}
            {plotSubtitle}
            """.split(
            "\n"
        )
    )

    plt.title(title)
    # patch parameter bug: https://github.com/slundberg/shap/issues/2362
    to_pass = SimpleNamespace(
        **{
            "values": localExplanations[k].values[:, 1]
            if len(localExplanations[k].values.shape) > 1
            else localExplanations[k].values,
            "data": localExplanations[k].data,
            "display_data": None,
            "feature_names": localExplanations.feature_names,
            "base_values": localExplanations[k].base_values[currentLabel]
            if len(localExplanations[k].base_values.shape) == 1
            else localExplanations[k].base_values,
        }
    )
    shap.plots.waterfall(to_pass, show=False)
    plt.tight_layout()
    plt.close(waterfallPlot)

    if config["tracking"]["remote"]:
        if config["tracking"]["remote"]:
            runTracker = neptune.init_run(
                project=f'{config["tracking"]["entity"]}/{config["tracking"]["project"]}',
                with_id=runID,
                api_token=config["tracking"]["token"],
                capture_stdout=False,
            )
        if not holdout:
            logPath = f"plots/samples/{j+1}/{sampleID}"
        else:
            logPath = f"plots/samples/holdout/{j+1}/{sampleID}"
        try:
            runTracker[logPath] = waterfallPlot
        except Exception:
            runTracker[logPath] = f"""failed to plot: {traceback.format_exc()}"""
    else:
        runPath = runID
        if not holdout:
            samplePlotPath = (
                f"{runPath}/featureImportance/shapelyExplanations/samples/{j+1}"
            )
            os.makedirs(samplePlotPath, exist_ok=True)
            waterfallPlot.savefig(
                f"{samplePlotPath}/{sampleID}.svg",
                bbox_inches="tight",
            )
        else:
            samplePlotPath = (
                f"{runPath}/featureImportance/shapelyExplanations/samples/holdout/{j+1}"
            )
            os.makedirs(samplePlotPath, exist_ok=True)
            waterfallPlot.savefig(
                f"{samplePlotPath}/{sampleID}.svg",
                bbox_inches="tight",
            )


def trackVisualizations(
    runID, plotSubtitle, modelName, current, holdout=False, config=config
):
    aucName = "aucPlot" if not holdout else "aucPlotHoldout"
    probabilities = (
        current["probabilities"] if not holdout else current["holdoutProbabilities"]
    )
    predictions = (
        current["predictions"] if not holdout else current["holdoutPredictions"]
    )
    labels = current["testLabels"] if not holdout else current["holdoutLabels"]
    ids = current["testIDs"] if not holdout else current["holdoutIDs"]
    labelsProbabilitiesByFold = {
        f"Fold {k+1}": (
            labels[k],
            np.array(probabilities[k])[:, 1],
        )
        if len(probabilities[k][0].shape) >= 1
        else (labels[k], probabilities[k])
        for k in range(config["sampling"]["crossValIterations"])
    }
    labelsPredictionsByFold = {
        f"Fold {k+1}": (labels[k], predictions[k])
        for k in range(config["sampling"]["crossValIterations"])
    }

    aucPlot = plotAUC(
        f"""
            Receiver Operating Characteristic (ROC) Curve
            {modelName} with {config['sampling']['crossValIterations']}-fold cross-validation
            {plotSubtitle}
            """,
        labelsProbabilitiesByFold,
        config=config,
    )
    confusionMatrixName = "confusionMatrix" if not holdout else "confusionMatrixHoldout"
    confusionMatrixList, avgConfusionMatrix = plotConfusionMatrix(
        f"""
            Confusion Matrix
            {modelName} with {config['sampling']['crossValIterations']}-fold cross-validation
            {plotSubtitle}
            """,
        labelsPredictionsByFold,
        config=config,
    )
    calibrationName = "calibrationPlot" if not holdout else "calibrationPlotHoldout"
    calibrationPlot = plotCalibration(
        f"""
            Calibration Curve
            {modelName} with {config['sampling']['crossValIterations']}-fold cross-validation
            {plotSubtitle}
            """,
        labelsProbabilitiesByFold,
        config=config,
    )
    if config["model"]["hyperparameterOptimization"] and not holdout:
        optimizerPlotName = "convergencePlot"
        optimizerPlot = plotOptimizer(
            f"""
                Hyperparameter convergence, mean squared error
                {modelName} with {config['sampling']['crossValIterations']}-fold cross-validation
                {plotSubtitle}
                """,
            {
                f"Fold {k+1}": [
                    result
                    for result in current["fittedOptimizer"][k].optimizer_results_
                ]
                for k in range(config["sampling"]["crossValIterations"])
            },
        )

    if (
        config["model"]["calculateShapelyExplanations"]
        and config["tracking"]["plotAllSampleImportances"]
    ):
        args = []
        for j in range(config["sampling"]["crossValIterations"]):
            for k in range(len(ids[j])):
                args.append(
                    (j, k, runID, modelName, plotSubtitle, current, holdout, config)
                )

        for arg in args:
            plotSample(*arg)

        # with multiprocess.Pool(multiprocess.cpu_count()) as pool:
        #     # Use map to apply the function to each argument set in the args list
        #     pool.starmap(plotSample, args)

        gc.collect()

    if config["tracking"]["remote"]:
        runTracker = neptune.init_run(
            project=f'{config["tracking"]["entity"]}/{config["tracking"]["project"]}',
            with_id=runID,
            api_token=config["tracking"]["token"],
            capture_stdout=False,
        )
        runTracker[f"plots/{aucName}"] = aucPlot
        for i, confusionMatrix in enumerate(confusionMatrixList):
            runTracker[f"{confusionMatrixName}/{i+1}"].upload(confusionMatrix)
        runTracker[
            f"average{confusionMatrixName[0].upper() + confusionMatrixName[1:]}"
        ].upload(avgConfusionMatrix)
        runTracker[f"plots/{calibrationName}"] = calibrationPlot
        if config["model"]["hyperparameterOptimization"] and not holdout:
            runTracker[f"plots/{optimizerPlotName}"] = optimizerPlot

        runTracker.stop()

    else:  # store plots locally
        runPath = runID
        aucPlot.savefig(f"{runPath}/{aucName}.svg", bbox_inches="tight")
        aucPlot.savefig(f"{runPath}/{aucName}.png", bbox_inches="tight")
        confusionMatrixPath = f"{runPath}/{confusionMatrixName}"
        os.makedirs(confusionMatrixPath, exist_ok=True)
        for i, confusionMatrix in enumerate(confusionMatrixList):
            confusionMatrix.savefig(
                f"{confusionMatrixPath}/{i+1}.svg", bbox_inches="tight"
            )
        avgConfusionMatrix.savefig(
            f"{runPath}/average{confusionMatrixName[0].upper() + confusionMatrixName[1:]}.svg",
            bbox_inches="tight",
        )
        avgConfusionMatrix.savefig(
            f"{runPath}/average{confusionMatrixName[0].upper() + confusionMatrixName[1:]}.png",
            bbox_inches="tight",
        )
        calibrationPlot.savefig(f"{runPath}/{calibrationName}.svg", bbox_inches="tight")
        calibrationPlot.savefig(f"{runPath}/{calibrationName}.png", bbox_inches="tight")
        if config["model"]["hyperparameterOptimization"] and not holdout:
            optimizerPlot.savefig(
                f"{runPath}/{optimizerPlotName}.svg", bbox_inches="tight"
            )
            optimizerPlot.savefig(
                f"{runPath}/{optimizerPlotName}.png", bbox_inches="tight"
            )

    plt.close("all")