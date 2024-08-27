#!usr/bin/python
# coding: utf-8

# Author: Dantae An
# License: MIT
# Description: MIDI 파일이 들어있는 디렉토리(폴더)를 입력받아 그 안에 있는
#              모든 MIDI 파일에 대해, 각 MIDI 곡에 들어있는 음표 정보를
#              추출하고 이를 "./output/" 폴더에 .json 파일로 출력합니다.

from mido import MidiFile, merge_tracks
import math

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