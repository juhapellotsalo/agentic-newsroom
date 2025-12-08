from agentic_newsroom.prompts.common import magazine_profile, magazine_guardrails, article_types

class NewsRoomContext:
    
    @staticmethod
    def build(agent_profile: str, custom_sections: str = "") -> str:
        """
        Creates a standard agentic newsroom prompt with the common headers
        plus the specific agent profile and any extra instructions.
        """
        ctx = f"""
            <Magazine Profile>
            This is the magazine you work for:
            {magazine_profile}
            </Magazine Profile>

            <Magazine Guardrails>
            Strictly adhere to the following editorial standards:
            {magazine_guardrails}
            </Magazine Guardrails>

            <Article Types>
            {article_types}
            </Article Types>

            <Your Profile>
            {agent_profile}
            </Your Profile>
            {custom_sections}
        """
        return ctx
