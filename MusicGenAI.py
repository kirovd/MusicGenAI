import numpy as np
from music21 import converter, instrument, note, chord
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Activation
from music21 import stream
import os

# Directory containing MIDI files
midi_dir = "/content/midi_songs"  # Update the path if necessary

if not os.path.exists(midi_dir):
    print(f"Error: The directory '{midi_dir}' does not exist.")
    exit()

# List all files in the directory
midi_files = [os.path.join(midi_dir, file) for file in os.listdir(midi_dir) if file.endswith(".mid")]

if not midi_files:
    print("Error: No MIDI files found in the 'midi_songs' folder.")
else:
    print(f"Found {len(midi_files)} MIDI files.")

    notes = []

    for file in midi_files:
        midi = converter.parse(file)
        print(f"Parsing {file}")

        notes_to_parse = None
        try:  # File has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() if s2 else midi.flat.notes
        except Exception as e:
            print(f"Error parsing {file}: {e}")
            continue

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    print(f"Total notes and chords found: {len(notes)}")

    # Prepare sequences for LSTM
    sequence_length = 100
    pitches = sorted(set(item for item in notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitches))

    # Create the reverse mapping from integers to notes
    int_to_note = {number: note for number, note in enumerate(pitches)}

    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(len(pitches))
    network_output = to_categorical(network_output)

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(256, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(256))
    model.add(Dense(len(pitches)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    # Train the model
    model.fit(network_input, network_output, epochs=10, batch_size=64)

    # Generate music
    start = np.random.randint(0, len(network_input) - 1)
    pattern = network_input[start]
    prediction_output = []

    for note_index in range(500):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction = model.predict(prediction_input, verbose=0)

        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern = np.append(pattern, index)
        pattern = pattern[1:len(pattern)]

    # Convert output to MIDI format
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = [note.Note(int(n)) for n in notes_in_chord]
            for n in notes:
                n.storedInstrument = instrument.Piano()
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='test_output.mid')
