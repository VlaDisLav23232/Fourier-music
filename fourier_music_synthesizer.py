import numpy as np
import pygame
import pygame.midi
import time
import colorsys
from collections import deque

class FourierMusicSynthesizer:
    def __init__(self):
        # Initialize pygame for audio and display
        pygame.init()
        pygame.midi.init()
        
        # Audio settings
        self.sample_rate = 44100  # Hz
        self.buffer_size = 1024
        self.channels = 2
        
        # Fourier synthesis parameters
        self.base_freq = 440.0  # A4 note
        self.harmonics = 10
        self.harmonic_weights = np.ones(self.harmonics) / np.arange(1, self.harmonics + 1)
        
        # Visualization settings
        self.width = 1200
        self.height = 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Fourier Music Synthesizer & Visualizer")
        
        # Load font and set font sizes
        self.main_font = pygame.font.SysFont('Arial', 20)
        self.title_font = pygame.font.SysFont('Arial', 24, bold=True)
        self.small_font = pygame.font.SysFont('Arial', 16)
        
        # Time tracking
        self.time_points = np.linspace(0, 2*np.pi, self.buffer_size)
        self.time_elapsed = 0
        self.last_time = time.time()
        
        # Data for visualization
        self.waveform_buffer = deque(maxlen=self.width)
        self.spectrum_buffer = np.zeros(self.buffer_size // 2)
        self.harmonic_display = np.zeros(self.harmonics)
        
        # Fill buffer with zeros initially
        for _ in range(self.width):
            self.waveform_buffer.append(0)
        
        # Colors for visualization
        self.colors = []
        for i in range(self.harmonics):
            hue = i / self.harmonics
            r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            self.colors.append((int(r*255), int(g*255), int(b*255)))
        
        # MIDI keyboard handling
        self.active_notes = {}
        self.key_to_note = {
            pygame.K_a: 60,  # C4
            pygame.K_w: 61,  # C#4
            pygame.K_s: 62,  # D4
            pygame.K_e: 63,  # D#4
            pygame.K_d: 64,  # E4
            pygame.K_f: 65,  # F4
            pygame.K_t: 66,  # F#4
            pygame.K_g: 67,  # G4
            pygame.K_y: 68,  # G#4
            pygame.K_h: 69,  # A4
            pygame.K_u: 70,  # A#4
            pygame.K_j: 71,  # B4
            pygame.K_k: 72,  # C5
            pygame.K_o: 73,  # C#5
            pygame.K_l: 74,  # D5
            pygame.K_p: 75,  # D#5
            pygame.K_SEMICOLON: 76,  # E5
        }
        
        # Current waveform profile
        self.current_profile = 'default'
        self.profile_names = {
            'default': 'Природне згасання гармонік (1/n)',
            'square': 'Прямокутна хвиля (непарні гармоніки)',
            'sawtooth': 'Пилкоподібна хвиля (всі гармоніки)',
            'triangle': 'Трикутна хвиля (непарні гармоніки з чергуванням)'
        }
        
        # Piano key colors
        self.piano_white_keys = ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ';']
        self.piano_black_keys = ['W', 'E', 'T', 'Y', 'U', 'O', 'P']
        self.piano_key_pressed = set()
        
        # Start audio stream
        self.mixer = pygame.mixer
        self.mixer.init(frequency=self.sample_rate, channels=self.channels)
        pygame.mixer.set_num_channels(32)  # Support for polyphony
        
        # Create empty sounds for each note
        self.cached_sounds = {}
        
        # Time tracking for live amplitude modulation
        self.start_time = time.time()
        self.amplitude_modulators = {}
        
        # Modulation settings
        self.mod_speed = 0.5  # Hz - speed of modulation 
        self.mod_depth = 0.3  # Depth of modulation (0-1)
        self.auto_modulation = True  # Toggle for automatic modulation
    
    def note_to_freq(self, note):
        # Convert MIDI note number to frequency (A4 = 69 = 440Hz)
        return 440.0 * (2 ** ((note - 69) / 12.0))
    
    def generate_fourier_tone(self, freq, duration=0.5, harmonic_profile='default'):
        """Generate a tone using Fourier synthesis with specified harmonic weights"""
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples, False)
        
        # Get weights based on harmonic profile
        if harmonic_profile == 'default':
            weights = np.ones(self.harmonics) / np.arange(1, self.harmonics + 1)
        elif harmonic_profile == 'square':
            # Only odd harmonics with 1/n falloff (square wave approximation)
            weights = np.zeros(self.harmonics)
            for i in range(self.harmonics):
                n = i + 1
                if n % 2 == 1:  # odd harmonics only
                    weights[i] = 1/n
        elif harmonic_profile == 'sawtooth':
            # All harmonics with 1/n falloff (sawtooth approximation)
            weights = np.array([1/(i+1) for i in range(self.harmonics)])
        elif harmonic_profile == 'triangle':
            # Odd harmonics with alternating sign and 1/n² falloff
            weights = np.zeros(self.harmonics)
            for i in range(self.harmonics):
                n = i + 1
                if n % 2 == 1:  # odd harmonics only
                    weights[i] = ((-1)**((n-1)//2)) / (n**2)
        else:
            weights = np.ones(self.harmonics) / np.arange(1, self.harmonics + 1)
        
        # Store current harmonic profile for visualization
        self.harmonic_display = weights
        
        # Generate tone from harmonics
        tone = np.zeros_like(t)
        for n in range(1, len(weights) + 1):
            # Add each harmonic with its weight
            amplitude = weights[n-1]
            tone += amplitude * np.sin(2 * np.pi * n * freq * t)
        
        # Apply envelope to avoid clicks
        envelope = np.ones_like(tone)
        attack = int(0.01 * self.sample_rate)
        release = int(0.01 * self.sample_rate)
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[-release:] = np.linspace(1, 0, release)
        tone = tone * envelope
        
        # Normalize
        tone = tone / np.max(np.abs(tone)) * 0.7
        
        # Convert to stereo - ensure C-contiguous array by creating a new array
        # instead of stacking
        stereo_tone = np.zeros((num_samples, 2), dtype=np.int16)
        stereo_tone[:, 0] = (tone * 32767).astype(np.int16)
        stereo_tone[:, 1] = (tone * 32767).astype(np.int16)
        
        # Update spectrum buffer for visualization
        self.spectrum_buffer = np.abs(np.fft.rfft(tone[:self.buffer_size]))
        
        # Update waveform buffer for visualization
        downsampled = tone[::4]  # Downsample for display
        for sample in downsampled[:self.width]:
            self.waveform_buffer.append(sample)
        
        return stereo_tone
    
    def get_time_modulated_amplitude(self, note):
        """Get amplitude modulated by time for a given note"""
        current_time = time.time()
        
        # Create a modulator for this note if it doesn't exist
        if note not in self.amplitude_modulators:
            # Different notes get slightly different speeds to create interesting patterns
            note_offset = (note % 12) / 24
            self.amplitude_modulators[note] = {
                'start_time': current_time,
                'speed': self.mod_speed * (1.0 + note_offset),
                'phase': np.random.random() * 2 * np.pi  # Random starting phase
            }
        
        mod_info = self.amplitude_modulators[note]
        elapsed = current_time - mod_info['start_time']
        
        if self.auto_modulation:
            # Sinusoidal modulation of amplitude
            mod_factor = 1.0 - self.mod_depth + self.mod_depth * np.sin(
                2 * np.pi * mod_info['speed'] * elapsed + mod_info['phase'])
            return max(0.3, mod_factor)  # Keep minimum volume at 30%
        else:
            return 0.7  # Default amplitude when modulation is off
    
    def update_modulation_params(self):
        """Update modulation based on time for interesting effects"""
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Slowly evolve modulation speed over time
        self.mod_speed = 0.5 + 0.3 * np.sin(0.1 * elapsed)
        
        # Every few seconds, slightly adjust modulation depth
        self.mod_depth = 0.3 + 0.2 * np.sin(0.05 * elapsed)
    
    def play_note(self, note):
        freq = self.note_to_freq(note)
        
        # Create unique key for each note + profile combination
        cache_key = (note, self.current_profile)
        
        # Generate or retrieve from cache
        if cache_key not in self.cached_sounds:
            stereo_tone = self.generate_fourier_tone(
                freq, duration=0.5, harmonic_profile=self.current_profile)
            # Ensure array is C-contiguous before creating sound
            if not stereo_tone.flags.c_contiguous:
                stereo_tone = np.ascontiguousarray(stereo_tone)
            sound = pygame.sndarray.make_sound(stereo_tone)
            self.cached_sounds[cache_key] = sound
        else:
            sound = self.cached_sounds[cache_key]
        
        # Apply time-based amplitude modulation
        amplitude = self.get_time_modulated_amplitude(note)
        
        # Play the sound with the modulated amplitude
        sound.set_volume(amplitude)
        sound.play(loops=0, maxtime=0, fade_ms=10)
        
        # Add to the set of currently pressed keys
        key_char = next((k for k, v in self.key_to_note.items() if v == note), None)
        if key_char:
            self.piano_key_pressed.add(pygame.key.name(key_char).upper())
    
    def stop_note(self, note):
        key_char = next((k for k, v in self.key_to_note.items() if v == note), None)
        if key_char:
            key_name = pygame.key.name(key_char).upper()
            if key_name in self.piano_key_pressed:
                self.piano_key_pressed.remove(key_name)
    
    def draw_waveform(self):
        """Draw the current waveform with improved labeling"""
        section_height = 200
        section_y = 100
        
        # Section title
        title = self.title_font.render("Форма хвилі (часова область)", True, (220, 220, 220))
        self.screen.blit(title, (20, section_y - 40))
        
        # Description
        description = self.small_font.render(
            "Показує зміну амплітуди звуку в часі, сформовану сумою всіх гармонік", 
            True, (180, 180, 180))
        self.screen.blit(description, (20, section_y - 20))
        
        # Draw section background
        pygame.draw.rect(self.screen, (30, 30, 40), 
                        (0, section_y - section_height//2, 
                         self.width, section_height))
        
        # Draw axis labels
        time_label = self.small_font.render("Час →", True, (180, 180, 180))
        self.screen.blit(time_label, (self.width - 50, section_y))
        
        amp_label = self.small_font.render("Амплітуда", True, (180, 180, 180))
        self.screen.blit(amp_label, (10, section_y - section_height//2 + 10))
        
        # Draw central line (zero amplitude)
        pygame.draw.line(self.screen, (100, 100, 100), 
                        (0, section_y), (self.width, section_y), 1)
        
        # Draw waveform
        points = []
        for x, sample in enumerate(self.waveform_buffer):
            y = section_y - int(sample * section_height // 2)
            points.append((x, y))
        
        if len(points) > 1:
            pygame.draw.aalines(self.screen, (0, 255, 200), False, points)
    
    def draw_spectrum(self):
        """Draw FFT spectrum with improved labeling"""
        section_height = 200
        section_y = 350
        
        # Section title
        title = self.title_font.render("Спектр частот (частотна область)", True, (220, 220, 220))
        self.screen.blit(title, (20, section_y - 40))
        
        # Description
        description = self.small_font.render(
            "Показує амплітуди окремих частот у звуці, отримані за допомогою перетворення Фур'є", 
            True, (180, 180, 180))
        self.screen.blit(description, (20, section_y - 20))
        
        # Draw section background
        pygame.draw.rect(self.screen, (30, 30, 40), 
                        (0, section_y - section_height, 
                         self.width, section_height))
        
        # Draw axis labels
        freq_label = self.small_font.render("Частота →", True, (180, 180, 180))
        self.screen.blit(freq_label, (self.width - 70, section_y - 10))
        
        mag_label = self.small_font.render("Амплітуда", True, (180, 180, 180))
        self.screen.blit(mag_label, (10, section_y - section_height + 10))
        
        # Draw frequency spectrum (logarithmic scale)
        max_val = np.max(self.spectrum_buffer) if np.max(self.spectrum_buffer) > 0 else 1
        normalized = self.spectrum_buffer / max_val
        
        # Only show first quarter of the spectrum (most interesting part)
        display_bins = min(self.width, len(normalized) // 4)
        
        for i in range(display_bins):
            # Logarithmic mapping for frequency bins
            x = int(self.width * np.log(i + 1) / np.log(display_bins))
            if x >= self.width:
                continue
                
            # Get magnitude and calculate height
            magnitude = normalized[i]
            height = int(magnitude * section_height)
            
            # Calculate color based on frequency
            hue = i / display_bins
            r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 0.9)
            color = (int(r*255), int(g*255), int(b*255))
            
            # Draw bar
            pygame.draw.rect(self.screen, color, 
                          (x, section_y - height, 
                           max(1, self.width/display_bins - 1), height))
    
    def draw_harmonics(self):
        """Draw current harmonic weights with improved labeling"""
        section_height = 170
        section_y = 600
        bar_width = 30
        
        # Section title
        title = self.title_font.render(
            f"Гармоніки - {self.profile_names[self.current_profile]}", 
            True, (220, 220, 220))
        self.screen.blit(title, (20, section_y - 40))
        
        # Description
        description = self.small_font.render(
            "Показує амплітуди кожної гармоніки, які формують звук", 
            True, (180, 180, 180))
        self.screen.blit(description, (20, section_y - 20))
        
        # Draw section background
        pygame.draw.rect(self.screen, (30, 30, 40), 
                        (0, section_y - section_height, 
                         self.width, section_height))
        
        # Draw harmonic weights
        max_weight = max(abs(self.harmonic_display)) if max(abs(self.harmonic_display)) > 0 else 1
        
        for i, weight in enumerate(self.harmonic_display):
            # Normalize weight
            norm_weight = weight / max_weight
            
            # Calculate bar height and position
            height = int(abs(norm_weight) * section_height * 0.8)
            x = 50 + i * (bar_width + 10)
            
            # Draw bar
            pygame.draw.rect(self.screen, self.colors[i], 
                          (x, section_y - height if norm_weight >= 0 else section_y, 
                           bar_width, height))
            
            # Draw harmonic number
            text = self.small_font.render(str(i+1), True, (200, 200, 200))
            self.screen.blit(text, (x + bar_width//2 - 5, section_y + 10))
        
        # Explain the concept of harmonics
        harmonic_note = self.small_font.render(
            "Кожна гармоніка - це частота, кратна основній частоті (1 = основна, 2 = вдвічі вища і т.д.)", 
            True, (180, 180, 180))
        self.screen.blit(harmonic_note, (20, section_y + 30))
    
    def draw_piano_keyboard(self):
        """Draw an interactive piano keyboard visualization"""
        white_key_width = 40
        white_key_height = 120
        black_key_width = 30
        black_key_height = 80
        
        keyboard_y = self.height - white_key_height - 20
        
        # Background for keyboard section
        pygame.draw.rect(self.screen, (40, 40, 50), 
                        (0, keyboard_y - 50, self.width, white_key_height + 70))
        
        # Keyboard title
        title = self.title_font.render("Клавіатура", True, (220, 220, 220))
        self.screen.blit(title, (20, keyboard_y - 40))
        
        # Draw white keys
        for i, key in enumerate(self.piano_white_keys):
            x = 300 + i * white_key_width
            
            # Check if key is pressed
            if key in self.piano_key_pressed:
                key_color = (180, 230, 255)
            else:
                key_color = (255, 255, 255)
            
            pygame.draw.rect(self.screen, key_color, 
                          (x, keyboard_y, white_key_width - 2, white_key_height))
            
            # Draw key label
            text = self.small_font.render(key, True, (0, 0, 0))
            self.screen.blit(text, (x + white_key_width//2 - 5, keyboard_y + white_key_height - 25))
        
        # Draw black keys
        black_key_positions = [0, 1, 3, 4, 5]  # Relative positions in an octave
        for i, key in enumerate(self.piano_black_keys):
            octave = i // 5
            position = i % 5
            
            x = 300 + (octave * 7 + black_key_positions[position]) * white_key_width + white_key_width - black_key_width//2
            
            # Check if key is pressed
            if key in self.piano_key_pressed:
                key_color = (100, 150, 200)
            else:
                key_color = (0, 0, 0)
            
            pygame.draw.rect(self.screen, key_color, 
                          (x, keyboard_y, black_key_width, black_key_height))
            
            # Draw key label
            text = self.small_font.render(key, True, (255, 255, 255))
            self.screen.blit(text, (x + black_key_width//2 - 5, keyboard_y + black_key_height - 25))
    
    def draw_controls(self):
        """Draw control instructions more clearly"""
        controls_x = 20
        controls_y = 20
        
        # Control panel background
        pygame.draw.rect(self.screen, (40, 40, 50, 200), 
                        (controls_x - 10, controls_y - 10, 270, 150), 
                        border_radius=5)
        
        # Title
        title = self.title_font.render("Керування:", True, (220, 220, 220))
        self.screen.blit(title, (controls_x, controls_y))
        
        # Waveform controls
        y_offset = controls_y + 30
        waveform_title = self.main_font.render("Зміна типу хвилі:", True, (200, 200, 200))
        self.screen.blit(waveform_title, (controls_x, y_offset))
        
        # Waveform buttons
        buttons = [
            ("1 - Стандартна", 'default'),
            ("2 - Прямокутна", 'square'),
            ("3 - Пилкоподібна", 'sawtooth'),
            ("4 - Трикутна", 'triangle')
        ]
        
        for i, (label, profile) in enumerate(buttons):
            y = y_offset + 25 + i * 25
            
            # Highlight current profile
            if profile == self.current_profile:
                pygame.draw.rect(self.screen, (60, 100, 150), 
                              (controls_x - 5, y - 2, 250, 24), 
                              border_radius=3)
            
            text = self.main_font.render(label, True, (220, 220, 220))
            self.screen.blit(text, (controls_x, y))
        
        # Add modulation control
        y_offset += 25 + len(buttons) * 25
        mod_text = self.main_font.render("M - Увімк/Вимк модуляцію амплітуди", True, (200, 200, 200))
        self.screen.blit(mod_text, (controls_x, y_offset))
        
        # Other controls
        other_controls_x = self.width - 300
        other_controls_y = 20
        
        # Background for other controls
        pygame.draw.rect(self.screen, (40, 40, 50, 200), 
                        (other_controls_x - 10, other_controls_y - 10, 290, 50), 
                        border_radius=5)
        
        # Exit control
        text = self.main_font.render("ESC - Вихід з програми", True, (220, 220, 220))
        self.screen.blit(text, (other_controls_x, other_controls_y))
    
    def draw_modulation_status(self):
        """Draw current modulation status and parameters"""
        status_x = self.width - 300
        status_y = 70
        
        # Status panel background
        pygame.draw.rect(self.screen, (40, 40, 50, 200), 
                        (status_x - 10, status_y - 10, 290, 130), 
                        border_radius=5)
        
        # Title
        title = self.title_font.render("Живе відстеження амплітуди:", True, (220, 220, 220))
        self.screen.blit(title, (status_x, status_y))
        
        # Status text
        status_text = "Увімкнено" if self.auto_modulation else "Вимкнено"
        status_color = (100, 255, 100) if self.auto_modulation else (255, 100, 100)
        
        status = self.main_font.render(f"Статус: {status_text}", True, status_color)
        self.screen.blit(status, (status_x, status_y + 30))
        
        # Modulation parameters
        speed_text = self.main_font.render(f"Швидкість модуляції: {self.mod_speed:.2f} Гц", 
                                          True, (220, 220, 220))
        self.screen.blit(speed_text, (status_x, status_y + 55))
        
        depth_text = self.main_font.render(f"Глибина модуляції: {self.mod_depth:.2f}", 
                                         True, (220, 220, 220))
        self.screen.blit(depth_text, (status_x, status_y + 80))
        
        # Draw modulation wave visualization
        wave_length = 250
        wave_height = 30
        wave_y = status_y + 110
        
        pygame.draw.rect(self.screen, (30, 30, 40), 
                       (status_x, wave_y, wave_length, wave_height))
        
        # Draw a sine wave representing current modulation
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        points = []
        for i in range(wave_length):
            x = status_x + i
            # Generate a sine wave with current modulation parameters
            t = elapsed + i / wave_length
            amplitude = np.sin(2 * np.pi * self.mod_speed * t)
            y = wave_y + wave_height//2 - int(amplitude * wave_height//2 * self.mod_depth)
            points.append((x, y))
        
        # Draw the modulation wave
        if len(points) > 1:
            pygame.draw.aalines(self.screen, (255, 200, 100), False, points)
    
    def draw_title(self):
        """Draw main application title"""
        # Title background
        pygame.draw.rect(self.screen, (50, 50, 70), 
                        (self.width//2 - 300, 15, 600, 40), 
                        border_radius=10)
        
        # Title text
        title_font = pygame.font.SysFont('Arial', 28, bold=True)
        title = title_font.render("Синтезатор та Візуалізатор Рядів Фур'є", True, (220, 220, 255))
        self.screen.blit(title, (self.width//2 - title.get_width()//2, 20))
    
    def run(self):
        """Main application loop"""
        running = True
        clock = pygame.time.Clock()
        
        try:
            while running:
                # Update time-based modulation parameters
                self.update_modulation_params()
                
                # Process events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    
                    # Handle key presses
                    elif event.type == pygame.KEYDOWN:
                        # Check for profile changes
                        if event.key == pygame.K_1:
                            self.current_profile = 'default'
                        elif event.key == pygame.K_2:
                            self.current_profile = 'square'
                        elif event.key == pygame.K_3:
                            self.current_profile = 'sawtooth'
                        elif event.key == pygame.K_4:
                            self.current_profile = 'triangle'
                        # Modulation control
                        elif event.key == pygame.K_m:
                            self.auto_modulation = not self.auto_modulation
                        # Exit on escape
                        elif event.key == pygame.K_ESCAPE:
                            running = False
                        # Notes
                        elif event.key in self.key_to_note:
                            note = self.key_to_note[event.key]
                            self.play_note(note)
                    
                    # Handle key releases
                    elif event.type == pygame.KEYUP:
                        if event.key in self.key_to_note:
                            note = self.key_to_note[event.key]
                            self.stop_note(note)
                
                # Clear screen
                self.screen.fill((20, 20, 30))
                
                # Draw main title
                self.draw_title()
                
                # Draw controls
                self.draw_controls()
                
                # Draw modulation status
                self.draw_modulation_status()
                
                # Draw visualizations
                self.draw_waveform()
                self.draw_spectrum()
                self.draw_harmonics()
                
                # Draw piano keyboard
                self.draw_piano_keyboard()
                
                # Update display
                pygame.display.flip()
                clock.tick(60)
        
        finally:
            # Clean up
            pygame.quit()

if __name__ == "__main__":
    app = FourierMusicSynthesizer()
    app.run()
