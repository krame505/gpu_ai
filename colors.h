
#ifndef _COLORS_H
#define _COLORS_H

#define RESET 0
#define BOLD 1
#define FAINT 2
#define ITALIC 3
#define UNDERLINE 4
#define BLINK 5
#define BLINK_RAPID 6
#define INVERSE 7
#define CONCEAL 8
#define STRIKETHROUGH 9
#define FONT(N) 1 << N
#define FRAKTUR 20
#define BOLD_OFF 21
#define FAINT_OFF 22
#define ITALIC_OFF 23
#define UNDERLINE_OFF 24
#define BLINK_OFF 25
#define INVERSE_OFF 27
#define CONCEAL_OFF 28
#define STRIKETHROUGH_OFF 29
#define FOREGROUND(N) 3 << N
#define BACKGROUND(N) 4 << N
#define FRAMED 51
#define ENCIRCLED 52
#define OVERLINED 53
#define FRAMED_OFF 54
#define OVERLINED_OFF 55

#define BLACK 0
#define RED 1
#define GREEN 2
#define YELLOW 3
#define BLUE 4
#define MAGENTA 5
#define CYAN 6
#define WHITE 7
#define DEFAULT 9

#define EFFECT(E) "\033[" << E << "m"

#endif
