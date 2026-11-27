Logos for the title slide.

Currently used:
  iisc_logo.png   -> shown CENTRED at the top of the title slide (official IISc seal)

The file can be .pdf, .png or .jpg. If it is missing, a dashed placeholder
shows in its place so the deck still compiles.

Want a second logo back (e.g. a department logo)?
  1. Drop system_logo.(pdf|png|jpg) here, and
  2. in ../presentation.tex, change the logo line to the two-logo row:
         \hbox to \textwidth{\hspace{0.5em}\showlogo{iisc_logo}{IISc}%
           \hfill\showlogo{system_logo}{System}\hspace{0.5em}}
     (the earlier two-logo version, kept in git history).

Tip: the logo height is set by \logoheight in ../presentation.tex (1.25 cm).
