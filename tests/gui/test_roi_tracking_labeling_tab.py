from whisker.gui.widgets import data_explorer
from whisker.gui.workflows.roi_tracking.tabs.labeling import RoiTrackingLabelingTab


class _FakeProject:
    name = "p1"


def _dataset_base_selection():
    return data_explorer.Selection(
        group=data_explorer.ItemGroupEnum.DATASETS,
        type=data_explorer.ItemTypeEnum.DATASET_BASE,
        item=("datasets", "ds1"),
    )


def _dataset_video_selection():
    return data_explorer.Selection(
        group=data_explorer.ItemGroupEnum.DATASETS,
        type=data_explorer.ItemTypeEnum.DATASET_VIDEO,
        item=("datasets", "ds1", "vid1.mp4"),
    )


def test_no_project_keeps_the_placeholder_shown(roi_workspace):
    ws, _video_path = roi_workspace
    tab = RoiTrackingLabelingTab()
    tab.set_workspace(ws)
    tab.on_data_explorer_item_selected(_dataset_video_selection())
    assert tab.view_stack.currentWidget() is tab.placeholder


def test_selecting_a_dataset_shows_the_summary(roi_workspace):
    ws, _video_path = roi_workspace
    tab = RoiTrackingLabelingTab()
    tab.set_workspace(ws)
    tab.set_project(_FakeProject())
    tab.on_data_explorer_item_selected(_dataset_base_selection())
    assert tab.view_stack.currentWidget() is tab.summary_widget
    assert tab.summary_widget.table.rowCount() == 1
    assert tab.summary_widget.table.item(0, 0).text() == "vid1.mp4"


def test_selecting_a_video_shows_the_verify_widget(roi_workspace):
    ws, _video_path = roi_workspace
    tab = RoiTrackingLabelingTab()
    tab.set_workspace(ws)
    tab.set_project(_FakeProject())
    tab.on_data_explorer_item_selected(_dataset_video_selection())
    assert tab.view_stack.currentWidget() is tab.verify_widget
    assert len(tab.verify_widget._tracks) == 2
    tab.verify_widget._autosave_timer.stop()


def test_summary_double_click_switches_to_the_verify_widget(roi_workspace):
    ws, video_path = roi_workspace
    tab = RoiTrackingLabelingTab()
    tab.set_workspace(ws)
    tab.set_project(_FakeProject())
    tab.on_data_explorer_item_selected(_dataset_base_selection())

    tab._on_summary_video_selected(video_path)
    assert tab.view_stack.currentWidget() is tab.verify_widget
    tab.verify_widget._autosave_timer.stop()


def test_switching_workspace_resets_to_the_placeholder(roi_workspace):
    ws, _video_path = roi_workspace
    tab = RoiTrackingLabelingTab()
    tab.set_workspace(ws)
    tab.set_project(_FakeProject())
    tab.on_data_explorer_item_selected(_dataset_base_selection())
    assert tab.view_stack.currentWidget() is tab.summary_widget

    tab.set_workspace(ws)
    assert tab.view_stack.currentWidget() is tab.placeholder
