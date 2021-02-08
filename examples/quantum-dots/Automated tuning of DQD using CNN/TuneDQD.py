from funcs import *


class TuneDQD:
    def __init__(
        self, patch_size, initial_voltages, n_avg, quantum_machine, **execute_args
    ):
        """

        @param patch_size: the patch size for scanning the voltages
        @type patch_size: int
        @param initial_voltages: the initial voltages for the plunger gates
        @type initial_voltages: tuple
        @param n_avg: the number for iterations to average
        @type n_avg: int
        @param quantum_machine: a QuantumMachine for execution
        @type quantum_machine:
        @param execute_args: execution arguments
        @type execute_args:
        """
        self.patch_size = patch_size
        self.lines_model = None
        self.transitions_model = None
        self.curr_voltages = initial_voltages
        self.curr_state = None
        self.n_avg = n_avg
        self.qm = quantum_machine
        self.execute_args = execute_args

    def generate_training_data(
        self,
        filename,
        gate_voltages,
        plunger_voltages,
        patches_per_diagram,
    ):
        """
        Generate data and labels for training
        @param filename: the file to save the data and labels to
        @type filename: str
        @param gate_voltages: gate voltages configurations
        @type gate_voltages: List[tuple]
        @param plunger_voltages: plunger gate voltages range and step: (v1_start, v1_end, v2_start, v2_end, step)
        @type plunger_voltages: tuple
        @param patches_per_diagram: number of patches per gate configuration
        @type patches_per_diagram: int
        @return:
        @rtype:
        """
        diagrams = generate_diagrams(
            gate_voltages, plunger_voltages, self.n_avg, self.qm, **self.execute_args
        )
        patches, lines_labels, transitions_labels = get_random_patches(
            diagrams, self.patch_size, patches_per_diagram
        )
        np.savez(filename, patches, lines_labels, transitions_labels)

    def train(self, filename, batch_size=20):
        """
        Trains and updates the models for distinguishing
        @param filename: the name of file containing the data and labels
        @type filename: str
        @param batch_size: batch size for training
        @type batch_size: int
        @return:
        @rtype:
        """
        mod1 = lines_model(self.patch_size)
        data = np.load(filename, allow_pickle=True)
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

    def find_zero(self, tries=10):
        """
        Algorithm step 1: find (0,0) charge state
        """

        if tries == 0:
            print(
                "Could not find the 0,0 state, try better training, different starting position, larger or smaller "
                "step size..."
            )
            return

        step = 0.03
        tolerance = 0.95

        # scan the local voltages
        patch = charge_stability_patch(
            self.curr_voltages[0] - self.patch_size * step,
            self.curr_voltages[0],
            self.curr_voltages[1] - self.patch_size * step,
            self.curr_voltages[1],
            step,
            self.n_avg,
            self.qm,
            **self.execute_args,
        )

        # predict whether the dots are empty or not
        prediction = self.lines_model.predict(np.array([patch]))
        if prediction[0] > tolerance:
            self.curr_state = (0, 0)
        elif prediction[1] > tolerance:
            self.curr_voltages = self.curr_voltages - self.patch_size * step
            self.find_zero(tries - 1)
        else:
            print("Could not resolve the state, need better accuracy")

    def tune_state(self, state):
        """
        Algorithm step 2: tune the DQD to be in 'state'
        :@param state: the desired target state
        :@type state: tuple
        """
        self.find_zero()
        if (self.curr_state != np.array([0, 0])).any():
            print("Could not find the 0,0 state")
            return

        step = 0.03
        i = 0
        max_tries = 50
        tolerance = 0.9
        while self.curr_state != state and i < max_tries:
            print(f"Step {i}/{max_tries}")
            i += 1
            patch = charge_stability_patch(
                self.curr_voltages[0],
                self.curr_voltages[0] + self.patch_size * step,
                self.curr_voltages[1],
                self.curr_voltages[1] + self.patch_size * step,
                step,
                self.n_avg,
                self.qm,
                **self.execute_args,
            )

            prediction = self.transitions_model.predict(np.array([patch]))
            trusted_prediction = [(p > tolerance).any() for p in prediction]
            pred_transition = [np.argmax(p) for p in prediction]
            diff = np.array(
                (state[0] - self.curr_state[0], state[1] - self.curr_state[1])
            )

            # make step in the right direction
            if diff[0] > 0 and diff[1] > 0:
                self.update_voltages([3, 2, 1], pred_transition)
            else:
                if diff[0] > 0:
                    self.update_voltages([1], pred_transition)
                if diff[1] > 0:
                    self.update_voltages([2], pred_transition)

        print(f"Reached state {self.curr_state}")

    def update_voltages(self, order, pred_transition):
        map_diff = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)}
        idx = None
        for o in order:
            try:
                idx = pred_transition.index(o)
                break
            except ValueError:
                pass
        if idx:
            self.curr_voltages += self.patch_size * np.array(map_diff[idx])
            self.curr_state += np.array(map_diff[idx])
