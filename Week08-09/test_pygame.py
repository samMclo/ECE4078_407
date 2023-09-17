import pygame

# Initialize Pygame
pygame.init()

# Set the window dimensions
screen_width = 800
screen_height = 800

# Create the Pygame window
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Robot Position Demo')

# Define colors
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)

# Robot attributes
robot_radius = 10
robot_position = (screen_width // 2, screen_height // 2)  # Starting position

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear the screen
    screen.fill(white)

    # Draw the robot
    pygame.draw.circle(screen, red, robot_position, robot_radius)

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
