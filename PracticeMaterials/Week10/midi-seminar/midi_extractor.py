#!usr/bin/python
# coding: utf-8

# Author: Dantae An
# License: MIT

from mido import MidiFile, merge_tracks, Message, MetaMessage, MidiTrack
from os import listdir, path, mkdir
import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import argparse
import sys

chord_templates = {}
OUT_DIR = './output/'
VERBOSE_SAVE = False  # True (save figures to .png files) or False (show figures without saving)
VERBOSE_MAX_INDEX = -1  # -1 (show all sequences to figures) or a positive integer (show # sequences to figures)

# 아래 배열은 조성 인식에 사용됩니다.
major_profile = [
  [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],  # C major
  [2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29],  # C# major
  [2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66],  # D major
  [3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39],  # D# major
  [2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19],  # E major
  [5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52],  # F major
  [2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09],  # F# major
  [4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38],  # G major
  [4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33],  # G# major
  [2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48],  # A major
  [3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23],  # A# major
  [2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35],  # B major
]
minor_profile = [
  [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],  # C minor
  [3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34],  # C# minor
  [3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69],  # D minor
  [2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98],  # D# minor
  [3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75],  # E minor
  [4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54],  # F minor
  [2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53],  # F# minor
  [3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60],  # G minor
  [2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38],  # G# minor
  [5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52],  # A minor
  [3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68],  # A# minor
  [2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33],  # B minor
]

unit_duration = 16  # 음표 길이 해상도: 16 (16nd note) or 32 (32nd note) 중 설정 가능!

unit_indices = [16, 8, 4, 2, 1]
unit_pow = 4
unit_divider = 4
if unit_duration == 32:
    unit_indices = [32, 16, 8, 4, 2, 1]
    unit_pow = 5
    unit_divider = 8

# 주어진 tick이 곡의 시작으로부터 unit_duration 기준으로
# 몇 번째 인덱스에 놓이는지 계산하여 반환
def unit_index(tick, tpb):
    # 항상 4/4박 가정
    unit_tick = tpb / unit_divider  # 16nd or 32nd note
    return math.floor(tick / unit_tick)

# 음 높이 값을 음이름으로 변환
def pitch_class(note_position):
    p = note_position % 12
    switcher = {
        0: 'C',
        1: 'C#',
        2: 'D',
        3: 'D#',
        4: 'E',
        5: 'F',
        6: 'F#',
        7: 'G',
        8: 'G#',
        9: 'A',
        10: 'A#',
        11: 'B'
    }
    return switcher.get(p)

# 하나의 MIDI 파일에 들어있는 모든 음표의 정보를 추출하고
# 음표의 시작 위치가 빠른 순으로 정렬하여 "notes"로 반환합니다.
# 반환 값은 (events, notes, ticks_per_beat) 입니다.
def parse_events(file):
    midi = MidiFile(file)
    return parse_events_helper(midi)

def parse_events_helper(midi):
    events = []
    notes = []
    single_note_on = []
    tick = 0
    timing = 0
    current_tempo = 500000  # default tempo (bpm 120)
    sequence_position = 0
    note_id = 0
    end_of_track = 0

    for msg in merge_tracks(midi.tracks):
        tp = msg.type
        is_meta = msg.is_meta
        delta_tick = msg.time
        if delta_tick <= 0:
            delta_tick = 0
        tick = tick + delta_tick
        timing = timing + round(delta_tick * current_tempo / midi.ticks_per_beat)
        attributes = []
        channel = None
        note_velocity = None
        note_position = None
        tempo = None
        numerator = None
        denominator = None

        clocks_per_click = None
        notated_32nd_notes_per_beat = None
        key = None
        control = None
        value = None
        program = None

        if tp == 'note_on' or msg.type == 'note_off':
            channel = msg.channel
            note_position = msg.note
            note_velocity = msg.velocity
            if note_velocity == 0:
                tp = 'note_off'
            attributes= ['Channel', 'Note_position', 'Note_velocity']
        elif tp == 'control_change':
            channel = msg.channel
            control = msg.control
            value = msg.value
            attributes = ['Channel', 'Control', 'Value']
        elif tp == 'program_change':
            channel = msg.channel
            program = msg.program
            attributes = ['Channel', 'Program']
        elif tp == 'time_signature':
            numerator = msg.numerator
            denominator = msg.denominator
            clocks_per_click = msg.clocks_per_click
            notated_32nd_notes_per_beat = msg.notated_32nd_notes_per_beat
            attributes = ['Numerator', 'Denominator', 'Clocks_per_click', 'Notated_32nd_notes_per_beat']
        elif tp == 'key_signature':
            key = msg.key
            # Valid values: A A#m Ab Abm Am B Bb Bbm Bm C C# C#m Cb Cm D D#m Db Dm E Eb Ebm Em F F# F#m Fm G G#m Gb Gm
            attributes = ['Key']
        elif tp == 'set_tempo':
            tempo = msg.tempo
            attributes = ['Tempo']
        elif tp == 'end_of_track':
            end_of_track = tick

        event = {
            'Type': tp,
            'Is_meta': is_meta,
            'Delta_tick': delta_tick,
            'Tick': tick,
            'Timing': timing,
            'Current_tempo': current_tempo,
            'Sequence_position': sequence_position,
            'Attributes': attributes,
            'Seq_index': unit_index(tick, midi.ticks_per_beat) // unit_duration,
            'Note_index': unit_index(tick, midi.ticks_per_beat) % unit_duration,
        }

        if channel is not None:
            event['Channel'] = channel
        if note_position is not None:
            event['Note_position'] = note_position
        if note_velocity is not None:
            event['Note_velocity'] = note_velocity
        if tempo is not None:
            event['Tempo'] = tempo
        if numerator is not None:
            event['Numerator'] = numerator
        if denominator is not None:
            event['Denominator'] = denominator
        if clocks_per_click is not None:
            event['Clocks_per_click'] = clocks_per_click
        if notated_32nd_notes_per_beat is not None:
            event['Notated_32nd_notes_per_beat'] = notated_32nd_notes_per_beat
        if key is not None:
            event['Key'] = key
        if control is not None:
            event['Control'] = control
        if value is not None:
            event['Value'] = value
        if program is not None:
            event['Program'] = program

        events.append(event)

        if tp == 'set_tempo':
            current_tempo = msg.tempo
        if tp == 'time_signature':
            current_numerator = msg.numerator
            current_denominator = msg.denominator

        sequence_position = sequence_position + 1

        if event['Type'] == 'note_on':
            # 'ID'는 임시 아이디이고, 나중에 새로 할당합니다.
            single_note_on.append({
                'ID': note_id,
                'Start_tick': event['Tick'],
                'End_tick': -1,
                'Start_timing': event['Timing'],
                'End_timing': -1,
                'Channel': event['Channel'],
                'Note_position': event['Note_position'],
                'Note_velocity': event['Note_velocity'],
                'Note_pitch_class': '',
                'Note_octave': -1,
                'Start_seq_index': unit_index(event['Tick'], midi.ticks_per_beat) // unit_duration,
                'Start_note_index': unit_index(event['Tick'], midi.ticks_per_beat) % unit_duration,
                'End_seq_index': -1,
                'End_note_index': -1,
                'Note_duration_units': 0
            })
            note_id = note_id + 1
        elif event['Type'] == 'note_off':
            try:
                note = next(s for s in single_note_on
                            if s['Channel'] == event['Channel'] and s['Note_position'] == event['Note_position'])
                single_note_on.remove(note)
                note['End_tick'] = event['Tick']
                note['End_timing'] = event['Timing']
                note['Note_pitch_class'] = pitch_class(event['Note_position'])
                note['Note_octave'] = (event['Note_position'] // 12) - 1
                note['End_seq_index'] = unit_index(event['Tick'], midi.ticks_per_beat) // unit_duration
                note['End_note_index'] = unit_index(event['Tick'], midi.ticks_per_beat) % unit_duration
                note['Note_duration_units'] = unit_index(event['Tick'], midi.ticks_per_beat) - unit_index(note['Start_tick'], midi.ticks_per_beat)
                notes.append(note)
            except StopIteration:
                pass

    notes.sort(key=(lambda e: e["Start_tick"]))

    return events, notes, midi.ticks_per_beat


# 디렉토리를 in_dir로 지정하면 해당 경로의 모든 MIDI 파일에 대해
# 음악적 특징을 추출하여 OUT_DIR 폴더에 파일로 저장합니다.
def read_dir(in_dir, verbose=False, one_hot=False):
    file_list = []
    mid_files = listdir(in_dir)

    ### valid (filtered) data (splitted), feature lists
    v_midi_list = []
    v_rhythm_density_list =[]
    v_note_density_list = []
    v_melodic_contour_list = []
    v_key_list = []
    v_global_key_list = []
    v_chord_list = []
    v_roman_numeral_chord_list = []
    v_note_velocity_list = []
    v_note_octave_list = []
    v_mean_note_pitch_list = []
    v_tempo_list = []
                          
    def is_midi_file(filename):
        ext = filename[filename.rindex('.'): len(filename)]
        return ext == '.mid' or ext == '.MID' or ext == '.midi' or ext == '.MIDI'

    mid_files = list(filter(is_midi_file, mid_files))
    n = len(mid_files)
    mid_files.sort()
    assert(n > 0)
    #print(mid_files)

    for i, in_path in enumerate(mid_files):
        fname = in_path.rstrip('.mid').rstrip('.MID').rstrip('.midi').rstrip('.MIDI')
        try:
            dup = next(d for d in file_list if d == fname)
            continue
        except StopIteration:
            pass

        print('Processing file {} of {}: {}'.format(i + 1, n, in_path))
        if not (in_dir[-1] == '\\' or in_dir[-1] == '/'):
            in_dir = in_dir + '/'

        try:
            events, notes, ticks_per_beat = \
                parse_events(in_dir.replace('\\', '/') + in_path)

            sequences, onset_sequences, sequences_length = \
                make_piano_roll_sequences(events, notes, ticks_per_beat, verbose)

            sequence_files, valid_sequences = sequences_to_midi_file(events, sequences_length, ticks_per_beat,
                                                                     in_dir.replace('\\', '/') + in_path, False)

            v_midi_list.extend(sequence_files)

            # extract_features의 인자로 valid_sequences에 numpy boolean array(shape=(sequence_length,))를 넘기면
            # 음표가 들어있지 않은 시퀀스를 자동으로 제외하고 low-level feature vector 배열을 생성하여
            # 그 길이가 sequence_files의 길이와 같게 된다.
            # 반면, None을 넘기면 음표가 들어있지 않은 시퀀스도 포함하여 그 길이가 sequence_length가 된다.
            features = extract_features(sequences, onset_sequences, sequences_length, events, verbose, one_hot,
                                        valid_sequences=valid_sequences)

            v_rhythm_density_list.extend(features['Rhythm_density'])
            v_note_density_list.extend(features['Note_density'])
            v_melodic_contour_list.extend(features['Melodic_contour'])
            v_key_list.extend(features['Key'])
            v_global_key_list.extend(features['Global_key'])
            v_chord_list.extend(features['Chord'])
            v_roman_numeral_chord_list.extend(features['Roman_numeral_chord'])
            v_note_velocity_list.extend(features['Note_velocity'])
            v_note_octave_list.extend(features['Note_octave'])
            v_mean_note_pitch_list.extend(features['Mean_note_pitch'])
            v_tempo_list.extend(features['Tempo'])

            assert(len(v_rhythm_density_list) == len(v_midi_list))
            assert(len(v_note_density_list) == len(v_midi_list))
            assert(len(v_melodic_contour_list) == len(v_midi_list))
            assert(len(v_key_list) == len(v_midi_list))
            assert(len(v_global_key_list) == len(v_midi_list))
            assert(len(v_chord_list) == len(v_midi_list))
            assert(len(v_roman_numeral_chord_list) == len(v_midi_list))
            assert(len(v_note_velocity_list) == len(v_midi_list))
            assert(len(v_note_octave_list) == len(v_midi_list))
            assert(len(v_mean_note_pitch_list) == len(v_midi_list))
            assert(len(v_tempo_list) == len(v_midi_list))
            
            data = {
                'Filename': fname,
                'Ticks_per_beat': ticks_per_beat,
                'Unit_duration': unit_duration,
                'Events': events,
                'Notes': notes,
                'Sequences': sequences,
                'Onset_sequences': onset_sequences,
                'Sequence_files': sequence_files,
                'Valid_sequences': valid_sequences,
                'Features': features
            }
            file_list.append(data)

        except IOError as e:
            print(e)
            continue

    if not path.exists(OUT_DIR+"extracted/"):
        mkdir(OUT_DIR+"extracted/")

    f = open(OUT_DIR+"extracted/metadata.txt", 'w')
    for i, name in enumerate(v_midi_list):
        f.write(str(i)+" "+name+"\n")
    f.close()

    v_rhythm_density_list = np.array(v_rhythm_density_list)
    v_note_density_list = np.array(v_note_density_list)
    v_melodic_contour_list = np.array(v_melodic_contour_list)
    v_key_list = np.array(v_key_list)
    v_global_key_list = np.array(v_global_key_list)
    v_chord_list = np.array(v_chord_list)
    v_roman_numeral_chord_list = np.array(v_roman_numeral_chord_list)
    v_note_velocity_list = np.array(v_note_velocity_list)
    v_note_octave_list = np.array(v_note_octave_list)
    v_mean_note_pitch_list = np.array(v_mean_note_pitch_list)
    v_tempo_list = np.array(v_tempo_list)

    print("Shapes for: Rhythm Density, Note Density, Melodic_contour, "
          "Key, Global_key, Chord, Roman_numeral_chord, Note_velocity, "
          "Note_octave, Mean_note_pitch, Tempo")
    print(v_rhythm_density_list.shape, v_note_density_list.shape, v_melodic_contour_list.shape,
          v_key_list.shape, v_global_key_list.shape, v_chord_list.shape, v_roman_numeral_chord_list.shape, v_note_velocity_list.shape,
          v_note_octave_list.shape, v_mean_note_pitch_list.shape, v_tempo_list.shape)

    np.save(OUT_DIR+"extracted/rhythm.npy", v_rhythm_density_list)
    np.save(OUT_DIR+"extracted/note_density.npy", v_note_density_list)
    np.save(OUT_DIR+"extracted/melodic_contour.npy", v_melodic_contour_list)
    np.save(OUT_DIR+"extracted/key.npy", v_key_list)
    np.save(OUT_DIR+"extracted/global_key.npy", v_global_key_list)
    np.save(OUT_DIR+"extracted/chord.npy", v_chord_list)
    np.save(OUT_DIR+"extracted/roman_numeral_chord.npy", v_roman_numeral_chord_list)
    np.save(OUT_DIR+"extracted/note_velocity.npy", v_note_velocity_list)
    np.save(OUT_DIR+"extracted/note_octave.npy", v_note_octave_list)
    np.save(OUT_DIR+"extracted/mean_note_pitch.npy", v_mean_note_pitch_list)
    np.save(OUT_DIR+"extracted/tempo.npy", v_tempo_list)

    return file_list

# Make sequences using piano roll representation
def make_piano_roll_sequences(events, notes, tpb, verbose=False):

    pitch_length = 128
    sequences_length = (unit_index(events[-1]["Tick"], tpb) // unit_duration) + 1

    # 피아노 롤 표현법으로 나타낼 시퀀스들의 배열 초기화
    sequences = np.zeros((sequences_length, unit_duration, pitch_length))

    # 시퀀스에서 각 음표의 시작 위치를 나타내는 배열 초기화
    onset_sequences = np.zeros((sequences_length, unit_duration, pitch_length))

    for note in notes:
        for i in range(note['Start_seq_index'], note['End_seq_index']):
            sequences[i,
                max(note['Start_seq_index'] * unit_duration + note['Start_note_index'], i * unit_duration) - (i * unit_duration):
                    min(note['End_seq_index'] * unit_duration + note['End_note_index'] + 1, (i + 1) * unit_duration) - (i * unit_duration),
                note['Note_position']
            ] = note['Note_velocity']
        if note['End_note_index'] > 0:
            sequences[note['End_seq_index'],
                max(note['Start_seq_index'] * unit_duration + note['Start_note_index'], note['End_seq_index'] * unit_duration) - (note['End_seq_index'] * unit_duration):
                    min(note['End_seq_index'] * unit_duration + note['End_note_index'] + 1, (note['End_seq_index'] + 1) * unit_duration) - (note['End_seq_index'] * unit_duration),
                note['Note_position']
            ] = note['Note_velocity']

        sequences[note['Start_seq_index'], note['Start_note_index'], note['Note_position']] = note['Note_velocity']
        onset_sequences[note['Start_seq_index'], note['Start_note_index'], note['Note_position']] = 1

    if verbose:
        # plot `sequences`
        plt.figure()
        visualizing_length = VERBOSE_MAX_INDEX
        if visualizing_length == -1:
            visualizing_length = sequences_length
        for i in range(0, visualizing_length):
            plt.subplot(2, visualizing_length, i + 1)
            plt.title("seq_{}".format(i))
            frame = plt.gca()
            frame.axes.get_xaxis().set_visible(False)
            frame.axes.get_yaxis().set_visible(False)
            plt.imshow(sequences[i, :, :].T, origin='lower')
            plt.subplot(2, visualizing_length, visualizing_length + i + 1)
            plt.title("onset_{}".format(i))
            frame = plt.gca()
            frame.axes.get_xaxis().set_visible(False)
            frame.axes.get_yaxis().set_visible(False)
            plt.imshow(onset_sequences[i, :, :].T, origin='lower')

        if VERBOSE_SAVE:
            plt.savefig(OUT_DIR + 'sequences.png', dpi=600)
        else:
            plt.show()

    return sequences, onset_sequences, sequences_length

# Make segmented token sequences to reconstruct midi files
def sequences_to_midi_file(events, sequences_length, tpb, original_filename, create_files=False):

    # 시퀀스에 들어있는 MIDI 이벤트 토큰들을 담는 배열 초기화
    token_sequences = []
    for i in range(sequences_length):
        token_sequences.append([])

    current_tempo = 500000
    current_seq_index = 0
    current_timing = 0

    for event in events:
        new_event = copy.deepcopy(event)

        if event['Tick'] // (tpb * 4) > current_seq_index:
            current_seq_index = event['Tick'] // (tpb * 4)
            current_timing = 0

            # 매 시퀀스의 시작마다 현재 템포를 설정하는 이벤트 삽입
            token_sequences[event['Tick'] // (tpb * 4)].append({
                'Type': 'set_tempo',
                'Is_meta': True,
                'Delta_tick': 0,
                'Tick': 0,
                'Timing': 0,
                'Current_tempo': current_tempo,
                'Sequence_position': -1,
                'Attributes': ['Tempo'],
                'Seq_index': current_seq_index,
                'Note_index': 0,
                'Tempo': current_tempo
            })

            # 매 시퀀스의 첫 이벤트의 delta tick 조정
            new_event['Delta_tick'] = event['Tick'] - tpb * 4 * current_seq_index

        new_event['Tick'] -= tpb * 4 * current_seq_index
        current_timing += round(new_event['Delta_tick'] * current_tempo / tpb)
        new_event['Timing'] = current_timing

        token_sequences[event['Tick'] // (tpb * 4)].append(new_event)

        if event['Type'] == 'set_tempo':
            current_tempo = event['Tempo']

    # 시퀀스 단위로 파일을 쪼개서 .mid로 저장하고 파일 이름 목록을 기록
    sequence_files = []

    # 음표가 하나 이상 들어있는(.mid 파일로 저장될) 시퀀스의 위치에 True를 기록
    valid_sequences = np.zeros(sequences_length, dtype=bool)

    if not path.exists(OUT_DIR):
        mkdir(OUT_DIR)

    for i in range(sequences_length):
        if create_files:
            new_midi = MidiFile(type=1)
            track = MidiTrack()
            new_midi.tracks.append(track)
            new_midi.ticks_per_beat = tpb

        single_note_on = []
        last_tick = 0
        has_note_on = False

        for event in token_sequences[i]:
            # 모든 이벤트를 빠짐없이 넣어야 타이밍이 어긋나지 않음
            tp = event['Type']
            if tp == 'note_on' and event['Note_velocity'] > 0:
                if create_files:
                    track.append(Message(tp, channel=event['Channel'], note=event['Note_position'],
                                        velocity=event['Note_velocity'], time=event['Delta_tick']))
                single_note_on.append((event['Note_position'], event['Channel']))
                has_note_on = True
            if tp == 'note_off' or (tp == 'note_on' and event['Note_velocity'] == 0):
                if create_files:
                    track.append(Message(tp, channel=event['Channel'], note=event['Note_position'],
                                        velocity=event['Note_velocity'], time=event['Delta_tick']))
                if (event['Note_position'], event['Channel']) in single_note_on:
                    single_note_on.remove((event['Note_position'], event['Channel']))
            elif tp == 'control_change':
                if create_files:
                    track.append(Message(tp, channel=event['Channel'], control=event['Control'],
                                        value=event['Value'], time=event['Delta_tick']))
            elif tp == 'set_tempo':
                if create_files:
                    track.append(MetaMessage(tp, tempo=event['Tempo'], time=event['Delta_tick']))
            elif tp == 'end_of_track':
                if create_files:
                    track.append(MetaMessage(tp, time=event['Delta_tick']))
            elif event['Is_meta']:
                if create_files:
                    track.append(MetaMessage(tp, time=event['Delta_tick']))

            last_tick = event['Tick']
        
        if create_files:
            for note in single_note_on:
                track.append(Message('note_off', channel=note[1], note=note[0],
                                        velocity=0, time=tpb * 4 - last_tick))
                last_tick = tpb * 4

        if create_files:
            track.append(MetaMessage('end_of_track', time=0))

        if has_note_on:
            filename = original_filename[original_filename.rfind('/') + 1:]
            if create_files:
                new_midi.save(OUT_DIR + str(i) + "_" + filename)
            sequence_files.append(str(i) + "_" + filename)
            valid_sequences[i] = True
    
    return sequence_files, valid_sequences


# Extract low-level features
def extract_features(sequences, onset_sequences, sequences_length, events, verbose=False, one_hot=False, valid_sequences=None):
    features = {}
    features['Rhythm_density'] = extract_rhythm_density(sequences, onset_sequences, sequences_length, verbose, one_hot,
                                                        valid_sequences=valid_sequences)
    features['Note_density'] = extract_note_density(sequences, sequences_length, verbose, one_hot,
                                                    valid_sequences=valid_sequences)
    features['Melodic_contour'] = extract_melodic_contour(sequences, onset_sequences, sequences_length, verbose, one_hot,
                                                          valid_sequences=valid_sequences)
    features['Key'] = extract_key(sequences, sequences_length, verbose, one_hot, valid_sequences=valid_sequences)
    features['Global_key'] = extract_global_key(sequences, sequences_length, verbose, one_hot, valid_sequences=valid_sequences)
    features['Chord'] = extract_chord(sequences, sequences_length, verbose, one_hot,
                                      valid_sequences=valid_sequences)
    features['Roman_numeral_chord'] = extract_roman_numeral_chord(sequences, sequences_length, verbose, one_hot,
                                                                  valid_sequences=valid_sequences)
    features['Note_velocity'] = extract_note_velocity(sequences, sequences_length, verbose, one_hot,
                                                      valid_sequences=valid_sequences)
    features['Note_octave'] = extract_note_octave(sequences, onset_sequences, sequences_length, verbose, one_hot,
                                                  valid_sequences=valid_sequences)
    features['Mean_note_pitch'] = extract_mean_note_pitch(sequences, onset_sequences, sequences_length, verbose,
                                                  valid_sequences=valid_sequences)
    features['Tempo'] = extract_tempo(events, sequences_length, verbose, one_hot, valid_sequences=valid_sequences)

    return features




# rhythm density
def extract_rhythm_density(sequences, onset_sequences, sequences_length, verbose=False, one_hot=False, valid_sequences=None):

    # low-level features 벡터 초기화
    rhythm_density_vector = np.zeros((sequences_length, unit_duration), dtype=np.int64)
    rhythm_density_vector_oh = np.zeros((sequences_length, unit_duration, 3))

    # rhythm_density
    temp_rhythm_density1 = np.max(sequences, axis=2)
    temp_rhythm_density2 = np.max(onset_sequences, axis=2)

    rhythm_density_vector_oh[temp_rhythm_density1 > 0] = [0, 0, 1]  # hold
    rhythm_density_vector_oh[temp_rhythm_density2 > 0] = [0, 1, 0]  # onset
    rhythm_density_vector_oh[(temp_rhythm_density1 <= 0) & (temp_rhythm_density2 <= 0)] = [1, 0, 0]  # rest

    if not one_hot:
        rhythm_density_vector[temp_rhythm_density1 > 0] = 2  # hold
        rhythm_density_vector[temp_rhythm_density2 > 0] = 1  # onset
        rhythm_density_vector[(temp_rhythm_density1 <= 0) & (temp_rhythm_density2 <= 0)] = 0  # rest

    if verbose:
        visualizing_length = VERBOSE_MAX_INDEX
        if visualizing_length == -1:
            visualizing_length = sequences_length
        plt.figure()
        for i in range(0, visualizing_length):
            if not (valid_sequences is not None and not valid_sequences[i]):
                plt.subplot(1, visualizing_length, i + 1)
                plt.title("rhythm_{}".format(i))
                frame = plt.gca()
                frame.axes.get_xaxis().set_visible(False)
                frame.axes.get_yaxis().set_visible(False)
                plt.imshow(rhythm_density_vector_oh[i, :, :].T, origin='lower')

        if VERBOSE_SAVE:
            plt.savefig(OUT_DIR + 'rhythm_density.png', dpi=600)
        else:
            plt.show()

    # 빈 시퀀스는 제외 -> 파일로 저장된 .mid 시퀀스의 수와 같아야 함
    if valid_sequences is not None:
        rhythm_density_vector = rhythm_density_vector[valid_sequences, ...]
        rhythm_density_vector_oh = rhythm_density_vector_oh[valid_sequences, ...]

    if one_hot:
        return rhythm_density_vector_oh
    else:
        return rhythm_density_vector


# note density
def extract_note_density(sequences, sequences_length, verbose=False, one_hot=False, valid_sequences=None):

    note_density_vector = np.clip(np.count_nonzero(sequences, axis=2), 0, 15)
    note_density_vector_oh = np.eye(16)[note_density_vector]

    if verbose:
        visualizing_length = VERBOSE_MAX_INDEX
        if visualizing_length == -1:
            visualizing_length = sequences_length
        plt.figure()
        for i in range(0, visualizing_length):
            if not (valid_sequences is not None and not valid_sequences[i]):
                plt.subplot(1, visualizing_length, i + 1)
                plt.title("note_{}".format(i))
                frame = plt.gca()
                frame.axes.get_xaxis().set_visible(False)
                frame.axes.get_yaxis().set_visible(False)
                plt.imshow(note_density_vector_oh[i, :, :].T, origin='lower')

        if VERBOSE_SAVE:
            plt.savefig(OUT_DIR + 'note_density.png', dpi=600)
        else:
            plt.show()
            
    if valid_sequences is not None:
        note_density_vector = note_density_vector[valid_sequences, ...]
        note_density_vector_oh = note_density_vector_oh[valid_sequences, ...]

    if one_hot:
        return note_density_vector_oh
    else:
        return note_density_vector


def monophony(onset_sequences, sequences, sequences_length):
    # using skyline algorithm to extract monophonic melody
    temp_monophony1 = onset_sequences.copy()
    temp_monophony2 = sequences.copy()
    # temp_old_pitch = -1

    # This algorithm cannot be fully vectorized!
    for i in range(0, sequences_length):
        """
        temp_old_pitch = -1
        temp_old_index = -1
        """
        for j in range(0, unit_duration):
            if j == 0 and len(np.nonzero(sequences[i, j, :])[0]) > 0:
                temp_monophony1[i, j, np.nonzero(sequences[i, j, :])[0][-1]] = 1
            temp_monophony1[i, j, np.nonzero(temp_monophony1[i, j, :])[0][0:-1]] = 0
            temp_monophony2[i, j, np.nonzero(sequences[i, j, :])[0][0:-1]] = 0
        """
            if len(np.nonzero(onset_sequences[i, j, :])[0]) > 0:
                temp_pitch = np.nonzero(onset_sequences[i, j, :])[0][-1]
                if temp_old_pitch == -1:
                    melodic_contour_vector_oh[i, temp_old_index + 1 : j + 1, :] = [0, 1, 0]  # 0 (첫 번째 음)
                    temp_old_pitch = temp_pitch
                elif temp_old_pitch < temp_pitch:
                    melodic_contour_vector_oh[i, temp_old_index + 1 : j + 1, :] = [0, 0, 1]  # + (직전 음보다 높은 음)
                    temp_old_pitch = temp_pitch
                elif temp_old_pitch == temp_pitch:
                    melodic_contour_vector_oh[i, temp_old_index + 1 : j + 1, :] = [0, 1, 0]  # 0 (직전 음과 같은 음)
                    temp_old_pitch = temp_pitch
                elif temp_old_pitch > temp_pitch:
                    melodic_contour_vector_oh[i, temp_old_index + 1 : j + 1, :] = [1, 0, 0]  # - (직전 음보다 낮은 음)
                    temp_old_pitch = temp_pitch
                temp_old_index = j

        melodic_contour_vector_oh[i, temp_old_index + 1: unit_duration, :] = [0, 1, 0]
        """
    return temp_monophony1, temp_monophony2

# melodic contour
def extract_melodic_contour(sequences, onset_sequences, sequences_length, verbose=False, one_hot=False, valid_sequences=None):

    melodic_contour_vector = np.zeros((sequences_length, unit_duration), dtype=np.int64)
    melodic_contour_vector_oh = np.zeros((sequences_length, unit_duration, unit_duration + 1))

    temp_monophony1, temp_monophony2 = monophony(onset_sequences, sequences, sequences_length)

    visualizing_length = VERBOSE_MAX_INDEX
    if visualizing_length == -1:
        visualizing_length = sequences_length
        
    if verbose:
        plt.figure()
        
    for i in range(0, sequences_length):
        temp_unique_pitch = np.unique(np.nonzero(temp_monophony1[i, :, :])[1])
        temp_unique_pitch = np.sort(temp_unique_pitch)

        # consider empty notes before the first note appears.
        temp_unique_pitch = np.insert(temp_unique_pitch, 0, -1)
        temp_old_pitch = -1

        for j in range(0, unit_duration):
            if len(np.nonzero(onset_sequences[i, j, :])[0]) > 0:
                melodic_contour_vector[i, j] = np.where(temp_unique_pitch == np.nonzero(temp_monophony1[i, j, :])[0])[0]
                melodic_contour_vector_oh[i, j, np.where(temp_unique_pitch == np.nonzero(temp_monophony1[i, j, :])[0])] = 1
                temp_old_pitch = np.nonzero(temp_monophony1[i, j, :])[0][0]
            else:
                melodic_contour_vector[i, j] = np.where(temp_unique_pitch == temp_old_pitch)[0]
                melodic_contour_vector_oh[i, j, np.where(temp_unique_pitch == temp_old_pitch)] = 1
                
        if verbose and not (valid_sequences is not None and not valid_sequences[i]) and i < visualizing_length:
            plt.subplot(2, visualizing_length, i + 1)
            plt.title("mono_{}".format(i))
            frame = plt.gca()
            frame.axes.get_xaxis().set_visible(False)
            frame.axes.get_yaxis().set_visible(False)
            plt.imshow(temp_monophony1[i, :, :].T, origin='lower')
            plt.subplot(2, visualizing_length, visualizing_length + i + 1)
            plt.title("contour_{}".format(i))
            frame = plt.gca()
            frame.axes.get_xaxis().set_visible(False)
            frame.axes.get_yaxis().set_visible(False)
            plt.imshow(melodic_contour_vector_oh[i, :, :].T, origin='lower')
            
            
    if verbose:
        if VERBOSE_SAVE:
            plt.savefig(OUT_DIR + 'melodic_contour.png', dpi=600)
        else:
            plt.show()
            

    if valid_sequences is not None:
        melodic_contour_vector = melodic_contour_vector[valid_sequences, ...]
        melodic_contour_vector_oh = melodic_contour_vector_oh[valid_sequences, ...]

    if one_hot:
        return melodic_contour_vector_oh
    else:
        return melodic_contour_vector


def extract_key(sequences, sequences_length, verbose=False, one_hot=False, valid_sequences=None):
    # http://rnhart.net/articles/key-finding/

    key_vector = np.zeros((sequences_length), dtype=np.int64)
    key_vector_oh = np.zeros((sequences_length, 25))

    eps = 0.0000000000000001

    visualizing_length = VERBOSE_MAX_INDEX
    if visualizing_length == -1:
        visualizing_length = sequences_length
        
    if verbose:
        plt.figure()
        
    for i in range(0, sequences_length):
        temp_unique_pitch, temp_pitch_count = np.unique(np.mod(np.nonzero(sequences[i, :, :])[1], 12), return_counts=True)
        temp_chroma = np.empty(12)
        temp_chroma.fill(eps)

        for k, p in enumerate(temp_unique_pitch):
            temp_chroma[p] += temp_pitch_count[k]

        max_corr = -1

        if len(temp_unique_pitch) > 1:
            for j in range(12):
                corr = np.corrcoef(temp_chroma, major_profile[j])[0][1]
                if max_corr < corr:
                    key_vector[i] = 1 + j
                    max_corr = corr
                corr = np.corrcoef(temp_chroma, minor_profile[j])[0][1]
                if max_corr < corr:
                    key_vector[i] = 13 + j
                    max_corr = corr

        key_vector_oh[i, key_vector[i]] = 1
        
        if verbose and not (valid_sequences is not None and not valid_sequences[i]) and i < visualizing_length:
            print(temp_chroma.astype(np.int64))
            if key_vector[i] == 0:
                print("no key")
            elif key_vector[i] <= 12:
                print(pitch_class(key_vector[i] - 1) + " major")
            else:
                print(pitch_class(key_vector[i] - 1) + " minor")

            plt.subplot(1, visualizing_length, i + 1)
            plt.title("key_{}".format(i))
            frame = plt.gca()
            frame.axes.get_xaxis().set_visible(False)
            frame.axes.get_yaxis().set_visible(False)
            plt.imshow(np.array([key_vector_oh[i, :]]).T, origin='lower')
            

    if verbose:
        if VERBOSE_SAVE:
            plt.savefig(OUT_DIR + 'key.png', dpi=600)
        else:
            plt.show()
            

    if valid_sequences is not None:
        key_vector = key_vector[valid_sequences, ...]
        key_vector_oh = key_vector_oh[valid_sequences, ...]

    if one_hot:
        return key_vector_oh
    else:
        return key_vector


def extract_global_key(sequences, sequences_length, verbose=False, one_hot=False, valid_sequences=None):

    global_key_vector = np.zeros((sequences_length), dtype=np.int64)
    global_key_vector_oh = np.zeros((sequences_length, 25))
    
    eps = 0.0000000000000001

    aggr_sequence = sequences.reshape(sequences.shape[0] * sequences.shape[1], sequences.shape[2])
    temp_unique_pitch, temp_pitch_count = np.unique(np.mod(np.nonzero(aggr_sequence[:, :])[1], 12), return_counts=True)
    temp_chroma = np.empty(12)
    temp_chroma.fill(eps)

    for k, p in enumerate(temp_unique_pitch):
        temp_chroma[p] += temp_pitch_count[k]

    max_corr = -1

    if len(temp_unique_pitch) > 1:
        for j in range(12):
            corr = np.corrcoef(temp_chroma, major_profile[j])[0][1]
            if max_corr < corr:
                global_key_vector.fill(1 + j)
                max_corr = corr
            corr = np.corrcoef(temp_chroma, minor_profile[j])[0][1]
            if max_corr < corr:
                global_key_vector.fill(13 + j)
                max_corr = corr

    global_key_vector_oh[:, global_key_vector[0]].fill(1)

    if valid_sequences is not None:
        global_key_vector = global_key_vector[valid_sequences, ...]
        global_key_vector_oh = global_key_vector_oh[valid_sequences, ...]

    if one_hot:
        return global_key_vector_oh
    else:
        return global_key_vector


# chord recognition using sliding window with size=(2nd note)
def extract_chord(sequences, sequences_length, verbose=False, one_hot=False, valid_sequences=None):

    chord_vector = np.zeros((sequences_length, unit_duration), dtype=np.int64)
    chord_vector_oh = np.zeros((sequences_length, unit_duration, 85))

    eps = 0.0000000000000001
    chord_types = ["maj", "min", "aug", "dim", "sus4", "dom7", "min7"]
    if not bool(chord_templates):
        # Execute only once
        chord_templates["maj"] = [[0.334 - 9 * eps, eps, eps, eps, 0.333, eps, eps, 0.333, eps, eps, eps, eps]]
        chord_templates["min"] = [[0.334 - 9 * eps, eps, eps, 0.333, eps, eps, eps, 0.333, eps, eps, eps, eps]]
        chord_templates["aug"] = [[0.334 - 9 * eps, eps, eps, eps, 0.333, eps, eps, eps, 0.333, eps, eps, eps]]
        chord_templates["dim"] = [[0.334 - 9 * eps, eps, eps, 0.333, eps, eps, 0.333, eps, eps, eps, eps, eps]]
        chord_templates["sus4"] = [[0.334 - 9 * eps, eps, eps, eps, eps, 0.333, eps, 0.333, eps, eps, eps, eps]]
        chord_templates["dom7"] = [[0.25 - 2 * eps, eps, eps, eps, 0.25 - 2 * eps, eps, eps, 0.25 - 2 * eps, eps, eps, 0.25 - 2 * eps, eps]]
        chord_templates["min7"] = [[0.25 - 2 * eps, eps, eps, 0.25 - 2 * eps, eps, eps, eps, 0.25 - 2 * eps, eps, eps, 0.25 - 2 * eps, eps]]
        for t in chord_types:
            for r in range(1, 12):
                temp_chord_template = chord_templates[t][0][12 - r: 12] + chord_templates[t][0][0: 12 - r]
                chord_templates[t].append(temp_chord_template)

    visualizing_length = VERBOSE_MAX_INDEX
    if visualizing_length == -1:
        visualizing_length = sequences_length
        
    if verbose:
        plt.figure()
        

    for i in range(0, sequences_length):
        if verbose and not (valid_sequences is not None and not valid_sequences[i]) and i < visualizing_length:
            print(str(i))

        for j in range(0, unit_duration):
            temp_frame = sequences[i, max(j - 4, 0): min(j + 3, 15), :]
            temp_unique_pitch, temp_pitch_count = np.unique(np.mod(np.nonzero(temp_frame)[1], 12), return_counts=True)
            temp_chroma = np.empty(12)
            temp_chroma.fill(eps)

            for k, p in enumerate(temp_unique_pitch):
                temp_chroma[p] += temp_pitch_count[k]

            if len(temp_unique_pitch) > 1:
                temp_chroma = np.divide(temp_chroma, np.sum(temp_chroma))
                temp_distance = np.empty(84)

                # Calculate KL Divergence
                for k, t in enumerate(chord_types):
                    temp_distance[12 * k: 12 * k + 12] = np.sum(np.add(np.subtract(np.multiply(chord_templates[t],
                                                                          np.log(np.divide(chord_templates[t],
                                                                                           temp_chroma))),
                                                              chord_templates[t]), temp_chroma), axis=1)
                temp_chord = np.argmin(temp_distance) + 1
                if verbose and not (valid_sequences is not None and not valid_sequences[i]) and i < visualizing_length:
                    # Print chord label
                    print(pitch_class((temp_chord - 1)), chord_types[(temp_chord - 1) // 12])
                chord_vector[i, j] = temp_chord
                chord_vector_oh[i, j, temp_chord] = 1
            else:
                # There is no chord (when # of unique pitch <= 1)
                if verbose and not (valid_sequences is not None and not valid_sequences[i]) and i < visualizing_length:
                    print("no chord")
                chord_vector[i, j] = 0
                chord_vector_oh[i, j, 0] = 1
                
        if verbose and not (valid_sequences is not None and not valid_sequences[i]) and i < visualizing_length:
            print()
            plt.subplot(1, visualizing_length, i + 1)
            plt.title("chord_{}".format(i))
            frame = plt.gca()
            frame.axes.get_xaxis().set_visible(False)
            frame.axes.get_yaxis().set_visible(False)
            plt.imshow(chord_vector_oh[i, :, :].T, origin='lower')
            

    if verbose:
        if VERBOSE_SAVE:
            plt.savefig(OUT_DIR + 'chord.png', dpi=600)
        else:
            plt.show()
            

    if valid_sequences is not None:
        chord_vector = chord_vector[valid_sequences, ...]
        chord_vector_oh = chord_vector_oh[valid_sequences, ...]

    if one_hot:
        return chord_vector_oh
    else:
        return chord_vector


# relative chord recognition from each measure
def extract_roman_numeral_chord(sequences, sequences_length, verbose=False, one_hot=False, valid_sequences=None):

    roman_numeral_chord_vector = np.zeros((sequences_length), dtype=np.int64)
    roman_numeral_chord_vector_oh = np.zeros((sequences_length, 85))

    temp_key_vector = extract_global_key(sequences, sequences_length, False, False, valid_sequences=None)

    eps = 0.0000000000000001
    chord_types = ["maj", "min", "aug", "dim", "sus4", "dom7", "min7"]
    if not bool(chord_templates):
        # Execute only once
        chord_templates["maj"] = [[0.334 - 9 * eps, eps, eps, eps, 0.333, eps, eps, 0.333, eps, eps, eps, eps]]
        chord_templates["min"] = [[0.334 - 9 * eps, eps, eps, 0.333, eps, eps, eps, 0.333, eps, eps, eps, eps]]
        chord_templates["aug"] = [[0.334 - 9 * eps, eps, eps, eps, 0.333, eps, eps, eps, 0.333, eps, eps, eps]]
        chord_templates["dim"] = [[0.334 - 9 * eps, eps, eps, 0.333, eps, eps, 0.333, eps, eps, eps, eps, eps]]
        chord_templates["sus4"] = [[0.334 - 9 * eps, eps, eps, eps, eps, 0.333, eps, 0.333, eps, eps, eps, eps]]
        chord_templates["dom7"] = [[0.25 - 2 * eps, eps, eps, eps, 0.25 - 2 * eps, eps, eps, 0.25 - 2 * eps, eps, eps, 0.25 - 2 * eps, eps]]
        chord_templates["min7"] = [[0.25 - 2 * eps, eps, eps, 0.25 - 2 * eps, eps, eps, eps, 0.25 - 2 * eps, eps, eps, 0.25 - 2 * eps, eps]]
        for t in chord_types:
            for r in range(1, 12):
                temp_chord_template = chord_templates[t][0][12 - r: 12] + chord_templates[t][0][0: 12 - r]
                chord_templates[t].append(temp_chord_template)

    def roman_numeral(relative_pitch, quality, key_major):
        p = relative_pitch % 12
        major_switcher = {
            0: 'I',
            1: '#I',
            2: 'II',
            3: '#II',
            4: 'III',
            5: 'IV',
            6: '#IV',
            7: 'V',
            8: '#V',
            9: 'VI',
            10: '#VI',
            11: 'VII',
            12: 'i',
            13: '#i',
            14: 'ii',
            15: '#ii',
            16: 'iii',
            17: 'iv',
            18: '#iv',
            19: 'v',
            20: '#v',
            21: 'vi',
            22: '#vi',
            23: 'vii'
        }
        minor_switcher = {
            0: 'I',
            1: '#I',
            2: 'II',
            3: 'III',
            4: '#III',
            5: 'IV',
            6: '#IV',
            7: 'V',
            8: 'VI',
            9: '#VI',
            10: 'VII',
            11: '#VII',
            12: 'i',
            13: '#i',
            14: 'ii',
            15: 'iii',
            16: '#iii',
            17: 'iv',
            18: '#iv',
            19: 'v',
            20: 'vi',
            21: '#vi',
            22: 'vii',
            23: '#vii'
        }

        switcher = minor_switcher
        if key_major:
            switcher = major_switcher
        
        if quality == "maj":
            return switcher.get(p)
        elif quality == "min":
            return switcher.get(p + 12)
        elif quality == "aug":
            return switcher.get(p) + str("+")
        elif quality == "dim":
            return switcher.get(p + 12) + str("o")
        elif quality == "sus4":
            return switcher.get(p) + str("sus4")
        elif quality == "dom7":
            return switcher.get(p) + str("7")
        else:  # quality == "min7":
            return switcher.get(p + 12) + str("7")

    visualizing_length = VERBOSE_MAX_INDEX
    if visualizing_length == -1:
        visualizing_length = sequences_length
        
    if verbose:
        plt.figure()
        

    for i in range(0, sequences_length):
        if verbose and not (valid_sequences is not None and not valid_sequences[i]) and i < visualizing_length:
            print(str(i))

        temp_frame = sequences[i, :, :]
        temp_unique_pitch, temp_pitch_count = np.unique(np.mod(np.subtract(np.nonzero(temp_frame)[1],
                                                                           (temp_key_vector[i] - 1) % 12), 12), return_counts=True)
        temp_chroma = np.empty(12)
        temp_chroma.fill(eps)

        for k, p in enumerate(temp_unique_pitch):
            temp_chroma[p] += temp_pitch_count[k]

        if len(temp_unique_pitch) > 1:
            temp_chroma = np.divide(temp_chroma, np.sum(temp_chroma))
            temp_distance = np.empty(84)

            # Calculate KL Divergence
            for k, t in enumerate(chord_types):
                temp_distance[12 * k: 12 * k + 12] = np.sum(np.add(np.subtract(np.multiply(chord_templates[t],
                                                                      np.log(np.divide(chord_templates[t],
                                                                                       temp_chroma))),
                                                          chord_templates[t]), temp_chroma), axis=1)
            temp_chord = np.argmin(temp_distance) + 1
            if verbose and not (valid_sequences is not None and not valid_sequences[i]) and i < visualizing_length:
                # Print chord label
                print(roman_numeral(temp_chord - 1, chord_types[(temp_chord - 1) // 12], temp_key_vector[i] < 13))
            roman_numeral_chord_vector[i] = temp_chord
            roman_numeral_chord_vector_oh[i, temp_chord] = 1
        else:
            # There is no chord (when # of unique pitch <= 1)
            if verbose and not (valid_sequences is not None and not valid_sequences[i]) and i < visualizing_length:
                print("no chord")
            roman_numeral_chord_vector[i] = 0
            roman_numeral_chord_vector_oh[i, 0] = 1
                
        if verbose and not (valid_sequences is not None and not valid_sequences[i]) and i < visualizing_length:
            print()
            plt.subplot(1, visualizing_length, i + 1)
            plt.title("roman_{}".format(i))
            frame = plt.gca()
            frame.axes.get_xaxis().set_visible(False)
            frame.axes.get_yaxis().set_visible(False)
            plt.imshow(roman_numeral_chord_vector_oh[i, :].T, origin='lower')
            

    if verbose:
        if VERBOSE_SAVE:
            plt.savefig(OUT_DIR + 'roman_numeral_chord.png', dpi=600)
        else:
            plt.show()
            

    if valid_sequences is not None:
        roman_numeral_chord_vector = roman_numeral_chord_vector[valid_sequences, ...]
        roman_numeral_chord_vector_oh = roman_numeral_chord_vector_oh[valid_sequences, ...]

    if one_hot:
        return roman_numeral_chord_vector_oh
    else:
        return roman_numeral_chord_vector


# velocity vector를 만들 때 각 시간 단위의 모든 음표의 velocity 평균을 구합니다. (0 제외)
# 이 함수는 one_hot일 때와 아닐 때 출력되는 값이 다릅니다!
def extract_note_velocity(sequences, sequences_length, verbose=False, one_hot=False, valid_sequences=None):

    note_velocity_vector = np.zeros((sequences_length, unit_duration), dtype=np.int64)
    note_velocity_vector_oh = np.zeros((sequences_length, unit_duration, 9))

    visualizing_length = VERBOSE_MAX_INDEX
    if visualizing_length == -1:
        visualizing_length = sequences_length
        
    if verbose:
        plt.figure()
        

    for i in range(0, sequences_length):
        for j in range(0, unit_duration):
            temp_max_velocity = np.max(sequences[i, j, :])
            if one_hot:
                note_velocity_vector[i, j] = (temp_max_velocity + 15) // 16
                note_velocity_vector_oh[i, j, int((temp_max_velocity + 15) // 16)] = 1
            else:
                note_velocity_vector[i, j] = temp_max_velocity
                
        if verbose and not (valid_sequences is not None and not valid_sequences[i]) and i < visualizing_length:
            plt.subplot(1, visualizing_length, i + 1)
            plt.title("velocity_{}".format(i))
            frame = plt.gca()
            frame.axes.get_xaxis().set_visible(False)
            frame.axes.get_yaxis().set_visible(False)
            plt.imshow(note_velocity_vector_oh[i, :, :].T, origin='lower')
            

    if verbose:
        if VERBOSE_SAVE:
            plt.savefig(OUT_DIR + 'note_velocity.png', dpi=600)
        else:
            plt.show()
            
    if valid_sequences is not None:
        note_velocity_vector = note_velocity_vector[valid_sequences, ...]
        if one_hot:
            note_velocity_vector_oh = note_velocity_vector_oh[valid_sequences, ...]

    if one_hot:
        return note_velocity_vector_oh
    else:
        return note_velocity_vector


# 0: no_note, 1 ~ 11: (highest_pitch // 12 + 1)
def extract_note_octave(sequences, onset_sequences, sequences_length, verbose=False, one_hot=False, valid_sequences=None):

    note_octave_vector = np.zeros((sequences_length, unit_duration), dtype=np.int64)
    # note_octave_vector_oh = np.zeros((sequences_length, unit_duration, 12))

    visualizing_length = VERBOSE_MAX_INDEX
    if visualizing_length == -1:
        visualizing_length = sequences_length
        
    if verbose:
        plt.figure()
        
    temp_monophony1, temp_monophony2 = monophony(onset_sequences, sequences, sequences_length)

    for i in range(0, sequences_length):
        temp_pitch = np.nonzero(temp_monophony2[i, :, :])

        for j, t in enumerate(temp_pitch[0]):
            note_octave_vector[i, t] = temp_pitch[1][j] // 12 + 1

    note_octave_vector_oh = np.eye(12)[note_octave_vector]
    
    if verbose:
        for i in range(0, sequences_length):
            if not (valid_sequences is not None and not valid_sequences[i]) and i < visualizing_length:
                plt.subplot(1, visualizing_length, i + 1)
                plt.title("octave_{}".format(i))
                frame = plt.gca()
                frame.axes.get_xaxis().set_visible(False)
                frame.axes.get_yaxis().set_visible(False)
                plt.imshow(note_octave_vector_oh[i, :, :].T, origin='lower')

    if verbose:
        if VERBOSE_SAVE:
            plt.savefig(OUT_DIR + 'note_octave.png', dpi=600)
        else:
            plt.show()
            
    if valid_sequences is not None:
        note_octave_vector = note_octave_vector[valid_sequences, ...]
        note_octave_vector_oh = note_octave_vector_oh[valid_sequences, ...]

    if one_hot:
        return note_octave_vector_oh
    else:
        return note_octave_vector


# 0: no_note, 0 ~ 127: mean(pitches)
# Note that one_hot option is not available!
def extract_mean_note_pitch(sequences, onset_sequences, sequences_length, verbose=False, valid_sequences=None):

    note_pitch_vector = np.zeros((sequences_length), dtype=np.float64)

    visualizing_length = VERBOSE_MAX_INDEX
    if visualizing_length == -1:
        visualizing_length = sequences_length
        
    if verbose:
        plt.figure()
        
    for i in range(0, sequences_length):
        temp_unique_pitch, temp_pitch_count = np.unique(np.nonzero(sequences[i, :, :])[1], return_counts=True)

        if np.sum(temp_pitch_count) > 0:
            for j, p in enumerate(temp_unique_pitch):
                note_pitch_vector[i] += p * temp_pitch_count[j] / np.sum(temp_pitch_count)
            
    if valid_sequences is not None:
        note_pitch_vector = note_pitch_vector[valid_sequences]

    return note_pitch_vector


# 이 함수는 one_hot일 때와 아닐 때 출력되는 값이 다릅니다!
def extract_tempo(events, sequences_length, verbose=False, one_hot=False, valid_sequences=None):
    tempo_vector = np.zeros((sequences_length, unit_duration), dtype=np.int64)
    #tempo_vector_oh = np.zeros((sequences_length, unit_duration, 7), dtype=np.int64)

    visualizing_length = VERBOSE_MAX_INDEX
    if visualizing_length == -1:
        visualizing_length = sequences_length
        
    if verbose:
        plt.figure()
        
    def tempo_bin(tempo):
        if tempo > 1200000:    # bpm < 50
            return 0
        elif tempo > 1000000:  # bpm < 60
            return 1
        elif tempo > 800000:   # bpm < 75
            return 2
        elif tempo > 600000:   # bpm < 100
            return 3
        elif tempo >= 444445:  # bpm < 135
            return 4
        elif tempo >= 333334:  # bpm < 180
            return 5
        else:                  # bpm > 180
            return 6

    temp_tempo = 500000
    temp_start_seq_index = 0
    temp_start_note_index = 0
    if one_hot:
        for e in events:
            if e['Type'] == 'set_tempo':
                temp_end_seq_index = e['Seq_index']
                temp_end_note_index = e['Note_index']

                if temp_start_seq_index == temp_end_seq_index:
                    tempo_vector[temp_start_seq_index, temp_start_note_index:temp_end_note_index] = tempo_bin(temp_tempo)
                else:
                    tempo_vector[temp_start_seq_index, temp_start_note_index:] = tempo_bin(temp_tempo)
                    tempo_vector[temp_start_seq_index + 1: temp_end_seq_index, :] = tempo_bin(temp_tempo)
                    tempo_vector[temp_end_seq_index, :temp_end_note_index] = tempo_bin(temp_tempo)

                temp_start_seq_index = temp_end_seq_index
                temp_start_note_index = temp_end_note_index
                temp_tempo = e['Tempo']

        tempo_vector[temp_start_seq_index, temp_start_note_index:] = tempo_bin(temp_tempo)
        if temp_start_seq_index + 1 < sequences_length:
            tempo_vector[temp_start_seq_index + 1:, :] = tempo_bin(temp_tempo)

        tempo_vector_oh = np.eye(7)[tempo_vector]
    else:
        for e in events:
            if e['Type'] == 'set_tempo':
                temp_end_seq_index = e['Seq_index']
                temp_end_note_index = e['Note_index']

                if temp_start_seq_index == temp_end_seq_index:
                    tempo_vector[temp_start_seq_index, temp_start_note_index:temp_end_note_index] = temp_tempo
                else:
                    tempo_vector[temp_start_seq_index, temp_start_note_index:] = temp_tempo
                    tempo_vector[temp_start_seq_index + 1: temp_end_seq_index, :] = temp_tempo
                    tempo_vector[temp_end_seq_index, :temp_end_note_index] = temp_tempo

                temp_start_seq_index = temp_end_seq_index
                temp_start_note_index = temp_end_note_index
                temp_tempo = e['Tempo']

        tempo_vector[temp_start_seq_index, temp_start_note_index:] = temp_tempo
        if temp_start_seq_index + 1 < sequences_length:
            tempo_vector[temp_start_seq_index + 1:, :] = temp_tempo


    if verbose:
        for i in range(0, sequences_length):
            if not (valid_sequences is not None and not valid_sequences[i]) and i < visualizing_length:
                plt.subplot(1, visualizing_length, i + 1)
                plt.title("tempo_{}".format(i))
                frame = plt.gca()
                frame.axes.get_xaxis().set_visible(False)
                frame.axes.get_yaxis().set_visible(False)
                plt.imshow(tempo_vector_oh[i, :, :].T, origin='lower')

    if verbose:
        if VERBOSE_SAVE:
            plt.savefig(OUT_DIR + 'tempo.png', dpi=600)
        else:
            plt.show()
            
    if valid_sequences is not None:
        tempo_vector = tempo_vector[valid_sequences, ...]
        if one_hot:
            tempo_vector_oh = tempo_vector_oh[valid_sequences, ...]

    if one_hot:
        return tempo_vector_oh
    else:
        return tempo_vector
    
# 직접 실행하면 지정한 폴더 내의 모든 MIDI 음악에 대해 음악적 특징들을 추출하여 파일로 저장합니다.
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose')
    parser.add_argument('-o', '--one_hot', action='store_true', help='apply one-hot encoding to feature vectors')
    parser.add_argument('-t', '--use_32nd_note', action='store_true', help='set note resolution to 32nd (default is 16th)')
    parser.add_argument('in_dirs', nargs='*', help='input directories that contain MIDI files')
    args = parser.parse_args()

    if len(args.in_dirs) == 1:
        if not path.exists(OUT_DIR):
            mkdir(OUT_DIR)

        print('Loading folder 1 of 1: {}'.format(sys.argv[1]))
        read_dir(args.in_dirs[0], args.verbose, args.one_hot)

    elif len(args.in_dirs) > 1:
        if not path.exists(OUT_DIR):
            mkdir(OUT_DIR)

        n = len(args.in_dirs)
        for i, d in enumerate(args.in_dirs):
            print('Loading folder {} of {}: {}'.format(i + 1, n, d))
            read_dir(d, args.verbose, args.one_hot)

    else:
        print('Usage:\n\tpython midi_extractor.py [-v] [-o] [-t] <in_dir_1> ... <in_dir_n>')