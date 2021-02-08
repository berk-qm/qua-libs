from funcs import *


class TuneDQD:
    def __init__(self, patch_size, initial_voltages):
        self.patch_size = patch_size
        self.lines_model = None
        self.transitions_model = None
        self.curr_voltages = initial_voltages
        self.curr_state = None

    def generate_training_data(
        self,
        path,
        gate_voltages,
        plunger_voltages,
        patches_per_diagram,
        quantum_machine,
        **execute_args
    ):
        diagrams = generate_diagrams(
            gate_voltages, plunger_voltages, quantum_machine, **execute_args
        )
        patches, lines_labels, transitions_labels = get_random_patches(
            diagrams, self.patch_size, patches_per_diagram
        )
        np.savez(path, patches, lines_labels, transitions_labels)

    def train(self, path, batch_size=20):
        mod1 = lines_model(self.patch_size)
        data = np.load(path, allow_pickle=True)
        images = data["arr_0"]
        lines_labels = data["arr_1"]
        transitions_labels = data["arr_2"]
        data.close()
        mod1.fit(images, lines_labels, batch_size=batch_size)
        self.lines_model = mod1

        # filter data to include only allowed transitions
        images_transitions = []
        labels_transitions = []
        for i, label in enumerate(transitions_labels):
            if label is not None:
                labels_transitions.append(label)
                images_transitions.append(images[i])

        mod2 = transitions_model(self.patch_size)
        labels_transitions = np.array(labels_transitions)
        mod2.fit(
            np.array(images_transitions),
            {
                "lower_right": labels_transitions[:, 0, :],
                "upper_left": labels_transitions[:, 1, :],
                "upper_right": labels_transitions[:, 2, :],
            },
            batch_size=batch_size,
            epochs=5,
        )
        self.transitions_model = mod2

    def find_zero(self):
        pass

    def tune_state(self, state):
        self.find_zero()
